# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile #unzip embedddings
import re
import difflib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
INPUT_PATH = "../input"
OUTPUT_PATH = "../output"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
CORRECTED_TEST_FILE = "corrected_test.csv"
CORRECTED_TRAIN_FILE = "corrected_train.csv"
GLOVE_EMBEDDING = "embeddings/glove.840B.300d/glove.840B.300d.txt"
#extract data from a csv file in a DataFrame
def load_data(csv_path = os.path.join(INPUT_PATH,TRAIN_FILE)):
    return pd.read_csv(csv_path)

#nested function returning True if len(X) is in [r_min, r_max] window, False otherwise
def len_in_range(r_min, r_max):
    def in_range(X):
        if len(X)>=r_min and len(X)<=r_max:
            return True
        else:
            return False
    return in_range

#nested function returning True if the prefix of oov (out of vocabulary) word 
#and voc word are the same, False otherwise
def prefix_comparator(oov, prefix_length = 1):
    def prefix_checker(voc):
        if voc[:prefix_length].lower() == oov[:prefix_length].lower():
            return True
        else:
            return False
    return prefix_checker

#compares oov_word to the vocabulary. If the similarity is over the ratio_treshold, the function will
#return the corresponding token in the vocabulary, otherwise the unknown_token value
#other parameters: 
#    hard_check: If hard_check is set to "False", function will replace ratio_treshold by 
#                min((len(oov_word)-1)/len(oov_word), ratio_threshold)

def oov_checker(oov_word, vocab_list, unknown_token = 'UNK', ratio_threshold = 0.9, hard_check = False):
    best_ratio = 0.
    best_voc = unknown_token    
    if hard_check:
        best_treshold = ratio_threshold
    else:
        best_treshold = min((len(oov_word)-1)/len(oov_word), ratio_threshold)
        
    for voc in vocab_list:
        ratio = difflib.SequenceMatcher(None, oov_word, voc).ratio()
        if ratio >= best_treshold and ratio > best_ratio:
            best_ratio = ratio
            best_voc = voc 

    return best_voc, best_ratio

#Generates a dict of tupples in which the keys are words in oov_list and the tupple contains
#the closest vocabulary word to the oov word and the associated score
#Only the words in vocabulary which 1.length is in [len(oov_word)-len_window, len(oov_word)+len_window]
#and 2. share the same prefix as oov_word are considered during the screening
def correction_score_generator(oov_list, vocabulary, unknown_token = 'UNK', len_window = 1):
    correction_list = {}
    if type(vocabulary) == list:
        vocab_list = vocabulary
    else:
        vocab_list = [*vocabulary]
        
    sorted_vocab_list = sorted(vocab_list, key=len)
    sorted_oov_list = sorted(oov_list, key=len)
    length = 0 
    for oov in sorted_oov_list:
        if length != len(oov):
            length = len(oov)
            min_len = length - len_window
            max_len = length + len_window
            vocab_window = len_in_range(min_len, max_len)
            filtered_vocab_list = list(filter(lambda X: vocab_window(X), sorted_vocab_list))
        prefix_comp = prefix_comparator(oov)
        filtered_vocab = list(filter(lambda X: prefix_comp(X), filtered_vocab_list))
        correction_list[oov] = oov_checker(oov, filtered_vocab, unknown_token, ratio_threshold = 0.)
    return correction_list

#Takes sentences list and return a corrected version of it based on a correction_dict and a treshold:
#1. scans each word of each sentence
#2. checks if the word is present in the embedding dict
#3. if not, checks if the word is present in the correction_dict and compares treshold with
#   the score of the proposed correction, if it exists
#4. if 2. and 3. are not positive, replace the unknown word by the "UNK" token
#5. returns the corrected sentence
def sentence_correcter(sentences, embeddings, correction_dict, threshold = 0.9):
    corrected_sentences = []
    unknown = "UNK"
    for sentence in sentences:
        corrected_sentence = []
        for word in sentence:
            if word in embeddings: 
                corrected_sentence.append(word)
            else:
                try:
                    if correction_dict[word][1] >= threshold:
                        corrected_sentence.append(correction_dict[word][0])
                    else:
                        corrected_sentence.append(unknown)
                except KeyError:
                    corrected_sentence.append(unknown)
        corrected_sentences.append(corrected_sentence)
    
    return corrected_sentences

def de_tokenize(sentences_list):
    de_sentences_list = []
    for sentence in sentences_list:
        de_sentence = ""
        for word in sentence:
            de_sentence += word + " "
        de_sentences_list.append(de_sentence)
        
    return de_sentences_list
#Only extract GloVe embedddings as a first approach
def glove_embeddings(gloveFile = os.path.join(INPUT_PATH, GLOVE_EMBEDDING), extract = -1):

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    
    embeddings_index = {}
    f = open(gloveFile,'r', encoding="utf8")
    increment = 0
    for line in f:
        word, vect = get_coefs(*line.split(" "))
        embeddings_index[word] = vect
        if increment == extract - 1:
            break
        elif extract != -1:
            increment += 1           
    return embeddings_index

#Returns a list of lists containing the tokenized version of the sentences contained in the sentences_list
def tokenize(sentences_list):
    return [re.findall(r"[\w]+|[']|[.,!?;]", x) for x in sentences_list]

#Return a dict containing as keys all unique words from a tokenized sentences list, and as value the 
#number of times these words appears in the sentences corpus
def get_vocab(sentences):
    """
    :param sentences: a list of list of words
    :return: a dictionary of words and their frequency 
    """
    vocab={}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] +=1
            except KeyError:
                vocab[word] = 1
    return vocab

#Finds words in common between a given embedding and the vocabulary
def compare_vocab_and_embeddings(vocab, embeddings_index):
    """
    :params vocab: our corpus vocabulary (a dictionary of word frquencies)
            embeddings_index: a genim object containing loaded embeddings.
    :returns in_common: words in common,
             in_common_freq: total frequency in the corpus vocabulary of 
                             all words in common
             oov: out of vocabulary words
             oov_frequency: total frequency in vocab of oov words
    """
    oov=[]
    in_common=[]
    in_common_freq = 0
    oov_freq = 0

    for word in vocab:
        if word in embeddings_index:
            in_common.append(word)
            in_common_freq += vocab[word]
        else: 
            oov.append(word)
            oov_freq += vocab[word]
    
    print('Found embeddings for {:.2%} of vocab'.format(len(in_common) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(in_common_freq / (in_common_freq + oov_freq)))

    return sorted(in_common)[::-1], sorted(oov)[::-1], in_common_freq, oov_freq

#Returns the list of out-of-vocabulary words sorted by their frequency 
# stated in the vocab dict
def sort_oov_words(oov, vocab, threshold = 0.5, min_len = 5):
    # Sort oov words by their frequency in the text
    sorted_oov= sorted(oov, key =lambda x: vocab[x], reverse=True )
    nr_tokens = 0
    i = 0
    ratio = 0.
    pruned_sorted_oov = []
    # Show oov words and their frequencies
    if (len(sorted_oov)>0):
        for word in sorted_oov:
            if len(word) >= min_len:
                if  re.search(r'[0-9]+', word, flags=0) == None:
                    nr_tokens +=vocab[word]
                    pruned_sorted_oov.append(word)
        print("Total number of oov instances: {}".format(nr_tokens))
        for word in pruned_sorted_oov:
            i += 1
            #print("%s\t%s"%(word, vocab[word]))
            ratio += vocab[word]
            if ratio/nr_tokens >= threshold:
                break       
    else:
        print("No words were out of vocabulary.")
    print("Number of oov words selected: {}/{} corresponding to {} instances".format(i, len(pruned_sorted_oov),ratio))  
    return pruned_sorted_oov[:i]
#extracts train data from train.csv
train_data = load_data()
train_data.head()
questions_list = train_data["question_text"].values
embeddings = glove_embeddings()
print("The GloVe embedding contains {} unique tokens".format(len(embeddings.keys())))
tokenized_questions = tokenize(questions_list)
token_dict = get_vocab(tokenized_questions)
print("The training dataset contains {} unique tokens".format(len(token_dict)))
in_common, oov, _, _ = compare_vocab_and_embeddings(token_dict, embeddings)
oov_words = sort_oov_words(oov, token_dict, threshold = 0.2)
correction_scored_dict = correction_score_generator(oov_words, embeddings)
corrected_questions = sentence_correcter(tokenized_questions, embeddings, correction_scored_dict, threshold = 0.8)
de_corrected_questions = de_tokenize(corrected_questions)
corrected_train_data = train_data.copy()
corrected_train_data['corrected_question_text'] = de_corrected_questions
corrected_train_data.to_csv(CORRECTED_TRAIN_FILE, index = False)
#extracts train data from train.csv
test_data = load_data(csv_path = os.path.join(INPUT_PATH,TEST_FILE))
test_data.head()
test_questions_list = test_data["question_text"].values
tokenized_test_questions = tokenize(test_questions_list)
token_test_dict = get_vocab(tokenized_test_questions)
print("The test dataset contains {} unique tokens".format(len(token_test_dict)))
test_in_common, test_oov, _, _ = compare_vocab_and_embeddings(token_test_dict, embeddings)
test_oov_words = sort_oov_words(test_oov, token_test_dict, threshold = 0.2)
test_correction_scored_dict = correction_score_generator(test_oov_words, embeddings)
corrected_test_questions = sentence_correcter(tokenized_test_questions, embeddings, test_correction_scored_dict, threshold = 0.8)
de_corrected_test_questions = de_tokenize(corrected_test_questions)
corrected_test_data = test_data.copy()
corrected_test_data['corrected_question_text'] = de_corrected_test_questions
corrected_test_data.to_csv(CORRECTED_TEST_FILE, index = False)