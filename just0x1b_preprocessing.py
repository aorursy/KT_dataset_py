import numpy as np

import pandas as pd



import nltk

from nltk.corpus import stopwords

from nltk.tokenize import wordpunct_tokenize



import csv



from gensim.models import word2vec

import gensim.models

# Read input data

train_data = pd.read_csv("../input/spooky-author-identification/train.csv")

test_data  = pd.read_csv("../input/spooky-author-identification/test.csv")



train_data.describe()


def preprocess_text(text, remove_list):   

    '''

    tokens = nltk.pos_tag(nltk.word_tokenize(text))

    print(tokens)

    good_words = [w for w, wtype in tokens if wtype not in remove_list]

    print(good_words)

    '''

    

    def clean_word(word):

        word = word.lower()

        if (len(word) > 1 and word[0] == '\''):

            return word[1:]

        return word

            

    tokens = nltk.pos_tag(nltk.word_tokenize(text))

    tokens = [(clean_word(word), pos) for (word,pos) in tokens]

    return [(word, pos) for (word,pos) in tokens if word not in remove_list]



def preprocess_corpus(corpus):

    

    stop_words = set(stopwords.words('english'))

    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 

    

    return [preprocess_text(text, stop_words) for text in corpus]



preprocessed_train_corpus = preprocess_corpus(train_data['text'])

preprocessed_test_corpus  = preprocess_corpus(test_data['text'])



print(preprocessed_train_corpus[0])
# GLOVE

def extract_vocabulary(corpus, vocabulary = set()):

    for sentence_chunk in corpus:

        for word,tag in sentence_chunk:

            vocabulary.add(word)

    return vocabulary

            

def extract_glove_mapping(vocabulary):   

    glove_table_path = '../input/glove.42b.300d/glove.42B.300d.txt'

    glove_table = pd.read_table(glove_table_path, sep=' ', index_col=0, header = None, quoting = csv.QUOTE_NONE)

    

    extracted_table = glove_table.loc[list(vocabulary)]

    

    del glove_table

    return extracted_table



vocabulary = extract_vocabulary(preprocessed_train_corpus)

vocabulary = extract_vocabulary(preprocessed_test_corpus, vocabulary)

glove_table = extract_glove_mapping(vocabulary)

glove_table.to_csv('glove_table.csv')
def glove_fill_missing(glove_table):

    # Find words with missing glove vecs

    missing_data = glove_table[glove_table.isnull().any(axis=1)]

    missing_words = list(missing_data.index)



    # Train Word2Vec from whole data

    sentences = []

    for sentence_chunk in preprocessed_train_corpus:

        sentences.append([word for (word, tag) in sentence_chunk])



    for sentence_chunk in preprocessed_test_corpus:

        sentences.append([word for (word, tag) in sentence_chunk])



    model = word2vec.Word2Vec(sentences,size=100, min_count=1)



    word2vec_model = model.wv

    del model

    

    # 

    def generate_glove_missing(missing_word,k = 5):

        similar_words = word2vec_model.similar_by_word(missing_word, topn=k)

        similar_words = [similar_word for (similar_word, similarity) in similar_words]

        glove_vec = np.nanmean(glove_table.loc[similar_words].as_matrix(),axis=0)

        return glove_vec

    

    for missing_word in missing_words:

        glove_vec = generate_glove_missing(missing_word)

        glove_table.loc[missing_word] = glove_vec

        

    return glove_table

    

glove_table =  glove_fill_missing(glove_table)
glove_table.to_csv('filled_glove_table.csv')

np.save("train_corp.npy", np.array(preprocessed_train_corpus))

np.save("test_corp.npy", np.array(preprocessed_test_corpus))