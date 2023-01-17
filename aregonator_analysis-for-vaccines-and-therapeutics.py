# These are all the imports that we need

import numpy as np # linear algebra

import pandas as pd # data processing

from nltk.tokenize.punkt import PunktSentenceTokenizer

import json

import os

from datetime import datetime

from os import listdir

from nltk.tokenize.regexp import regexp_tokenize

from nltk.corpus import stopwords

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

from nltk import pos_tag

from datetime import datetime

from string import digits

from string import printable

import multiprocessing

import string

import spacy as sp

import collections

import math

import random

from itertools import islice

from keras.models import Model

from keras.layers import Input, Dense, Reshape, merge, Lambda

from keras.layers.embeddings import Embedding

from keras.preprocessing.sequence import skipgrams

from keras.preprocessing import sequence

import tensorflow as tf

from keras import backend as K

import matplotlib.pyplot as plt

import multiprocessing as mp

from statistics import mean 

import random

import gc



# loading of the spacy dataset for the list of stopwords

nlp = sp.load("en_core_web_sm")
def configure_sentence_tokenizer():

    #configures the sentence tokenizer

    toeknizer = PunktSentenceTokenizer()

    abbrevations = ['Fig.', 'Table.','et al.', 'vs.', 'eg.', 'e.g.', 'Calcd.', 'r.p.m.', 'i.e.', 'i.m.', 'i.n.', 'pers.', 'spp.', 'p.i.', 'etc.', 'U.S.', 'i.n.']

    for abbrevation in abbrevations:

        toeknizer._params.abbrev_types.add(abbrevation)

    return toeknizer



def process_paper_json(dataframe_data, json_doc, tokenizer, paper_ids):

    #json_doc is the dictionary object representing the json

    #this function basicaly processes the json,

    #extracts the abstract and the text body

    #splits the abstract and body into sentences and paragraphs

    #and creates a dataframe for it

    

    #crateing the dataframe with the scheema

    #type can be 'title', 'abstract', 'text'

    columns = ['paper_id', 'paragraph_number', 'sentence_number', 'type', 'content']

    

    

    has_title = True

    has_text = True

    has_abstract = True

    if 'paper_id' in json_doc:

        paper_id = json_doc['paper_id']

        if (paper_id not in paper_ids):

            return

    else:

        print('ERROR paper id was not present')

        return

    if 'metadata' in json_doc:

        if 'title' in json_doc['metadata']:

            title = json_doc['metadata']['title']

        else:

            has_title = False

            print('ERROR no title for paper id: ' + str(paper_id))

    else:

        has_title = False

        print('ERROR no metadata for paper id: ' + str(paper_id))

    

    if 'abstract' in json_doc:

        abstract = json_doc['abstract']

    else:

        has_abstract = False

        #print('ERROR no abstract for paper id: ' + str(paper_id))

        

    if 'body_text' in json_doc:

        body_tect = json_doc['body_text']

    else:

        has_text = False

        #print('ERROR no text for  paper id: ' + str(paper_id))

    

    #going through the title

    if (has_title):

        sent_number = 0

        for sentence in tokenizer.tokenize(title):

            dataframe_data.append([paper_id, 0, sent_number, 'title', sentence])

            sent_number = sent_number + 1

    

    #going through the abstract

    if (has_abstract):

        sent_number = 0

        paragraph_number = 0

        for paragraph in abstract:

            paragraph_text = paragraph['text']

            paragraph_text = paragraph_text.replace('et al.', '')

            for sentence in tokenizer.tokenize(paragraph_text):

                dataframe_data.append([paper_id, paragraph_number, sent_number, 'abstract', sentence])

                sent_number = sent_number + 1

            paragraph_number = paragraph_number + 1

    

    #going through the body

    if (has_text):

        sent_number = 0

        paragraph_number = 0

        for paragraph in body_tect:

            paragraph_text = paragraph['text']

            paragraph_text = paragraph_text.replace('et al.', '')

            for sentence in tokenizer.tokenize(paragraph_text):

                dataframe_data.append([paper_id, paragraph_number, sent_number, 'text', sentence])

                sent_number = sent_number + 1

            paragraph_number = paragraph_number + 1

    

    return dataframe_data

    

    

    

    

#the nltk wordnet_lemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

#the final list of all the stopwords to be removed (combined from nltk, spacy and custom)

global stop_words



def get_all_stop_words(nlp):

    #getting a collection of stop words

    nlp = sp.load('en_core_web_sm')



    stop_words = []

    spacy_stopwords = nlp.Defaults.stop_words

    nltk_stop_words = set(stopwords.words('english')) 



    for stop_word in spacy_stopwords:

        if (stop_word not in stop_words):

            stop_words.append(stop_word)





    for stop_word in nltk_stop_words:

        if (stop_word not in stop_words):

            stop_words.append(stop_word)

    # custom list of stopwords that are specific to papers       

    stop_words = np.concatenate((stop_words, ['table',  'fig', 'figure', 'etc', 'appendix', 'file', 'citation']))

    return stop_words





def clean_up_and_tokenize_text(text):

    #given the text, converts to lower case

    # removes punctuation

    # removes the digits

    # removes non-english charachters

    # tokenizes the wods

    text = text.lower().translate(str.maketrans('', '', string.punctuation))

    text = text.translate(str.maketrans('', '', digits))

    text = ''.join(i for i in text if i in printable)

    # tokenizing

    tokens = regexp_tokenize(text, pattern = '\s+', gaps = True)

    #for index in range(len(tokens)):

    #    tokens[index] = wordnet_lemmatizer.lemmatize(tokens[index], wordnet.NOUN)

    return tokens

    

def clean_up_data(data):

    # main function for cleanup of the entire dataframe

    num_records = len(data)

    start_time = datetime.now()

    for data_index in range(num_records):

        clean_text = ''

        # tokenizing

        tokens = clean_up_and_tokenize_text(data.iat[data_index, 4])

        # removing stopwords

        tokens = [word for word in tokens if not word in stop_words]

        # lemmatizing in parralel

        #tags = pos_tag(tokens)

        if (len(tokens) <= 1):

            clean_text = ''

        else:

            for token in tokens:

                clean_text += token + ' '

        data.iat[data_index, 4] = clean_text

        if (data_index%10000 == 0):

            end_time = datetime.now()

            time_lapse = end_time - start_time

            print('working on ' + str(data_index) + 'th data point out of ' + str(num_records) + ' points. %' + str(data_index*1.0/num_records*100) + ' complete in ' + str(time_lapse.seconds) + 's')

    return data    







def get_dataframe_from_paper_ids(paper_ids):

    # given the list of paper_ids

    # iterates over all the files in the dataset

    # creates a data_frame, cleans up the dataframe

    # returns the clean dataframe

    

    columns = ['paper_id', 'paragraph_number', 'sentence_number', 'type', 'content']



    final_data = []

    tokenizer = configure_sentence_tokenizer()



    num_files = 0

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            num_files = num_files + 1

    print('Total number of files = ' + str(num_files))

    total_count = 0

    start_time = datetime.now()

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            if (filename.endswith('.json')):

                with open(os.path.join(dirname, filename), 'r') as f:

                    sample_json_dict = json.load(f)

                    process_paper_json(final_data, sample_json_dict, tokenizer, paper_ids)

                    total_count = total_count + 1

                    if (total_count%5000 == 0):

                        time_now = datetime.now()

                        elapsed_time = time_now - start_time

                        print('Processed: ' + str(total_count) + ' Documents remaining: ' 

                              + str(num_files - total_count) + ' done: ' 

                              + str((total_count*1.0)/(num_files)*100.0) + '%'

                                    + ' expected time: ' + str(num_files/total_count*elapsed_time.seconds)

                                    + 's'+ ' time: ' + str(elapsed_time.seconds) + 's')





    final_dataframe = pd.DataFrame(final_data, columns=columns)    

    print('dataframe created, dropping NAs ' + str(final_dataframe.shape))

    final_dataframe.dropna(inplace=True)

    

    stop_words = get_all_stop_words(nlp)

    start_time = datetime.now()

    final_dataframe = clean_up_data(final_dataframe)

    final_dataframe.dropna(inplace=True)

    end_time = datetime.now()

    time_lapse = end_time - start_time

    print('full data cleanup took: ' + str(time_lapse.seconds) + 's to clean' )



    return final_dataframe;

    #print(final_dataframe.head())

    #print('writing to file ' + str(final_dataframe.shape))

    #final_dataframe.to_csv (r'/kaggle/working/data.csv', index = False, header=True)

    #print('file has been created')

    



def get_list_of_words(data):

    # this function converts

    # all the strings in the dataframe to a sequential list of tokens

    num_records = len(data)

    #this is the data set of all the words

    word_list = []

    for index in range(num_records):

        text = data.iat[index, 4]

        tokens = regexp_tokenize(text, pattern = '\s+', gaps = True)

        word_list.extend(tokens)

    

    print(word_list[:10])

    return word_list



def build_dataset(word_set, n_words, keywords = []):

    #gets the most common set of words into a dictionary with indexes

    # it also adds the list of the keywords, in case the keywords are not in the most

    # common list

    count = [['UNK', -1]]

    all_word_counts = collections.Counter(word_set);

    most_common_words = all_word_counts.most_common(n_words)

    most_common_words_list = []

    

    all_counts = []

    for word, freq in all_word_counts.items():

        all_counts.append(freq)

    all_counts.sort(reverse=True)

    

    fig, ax = plt.subplots()

    ax.plot(range(len(all_counts)), all_counts)



    ax.set(xlabel='index', ylabel='count of words',

           title='The sorted count of words in the selected paper set')

    ax.grid()

    plt.show()

        

    

    for word in most_common_words:

        most_common_words_list.append(word[0])

    count.extend(most_common_words)

    

    #adding the keywords that are missing

    for keyword in keywords:

        if (keyword not in most_common_words_list) and (keyword in all_word_counts):

            count.append([keyword, all_word_counts[keyword]])

            print('adding keyword: {} to the list since it was not there'.format(keyword))

            

    for keyword in keywords:

        if (keyword not in all_word_counts):

            print('keyword {} is missing from the most common list'.format(keyword))

            

    dictionary = dict()

    for word, _ in count:

        dictionary[word] = len(dictionary)

    data = list()

    unk_count = 0

    for word in word_set:

        if word in dictionary:

            index = dictionary[word]

        else:

            index = 0 

            unk_count += 1

        data.append(index)

    count[0][1] = unk_count

    

    # reverse lookup

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, dictionary, reversed_dictionary, len(dictionary)





#the generic skipgram for the word embeddings tasks given by Keras

def get_skip_grams(data, vocab_size, keywords, dictionary):

    #generate the list of skipgrams to use for the training data

    

    window_size = 3

    start_time = datetime.now()



    sampling_table = sequence.make_sampling_table(vocab_size)

    # this is the skipgrams in keras

    couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)

    word_target, word_context = zip(*couples)

    word_target = np.array(word_target, dtype="int32")

    word_context = np.array(word_context, dtype="int32")

    

    training_sample_counts = collections.Counter(word_target);

    all_counts = []

    for word_index, count in training_sample_counts.items():

        all_counts.append(count)

        

    keyword_indexes_list = []

    for keyword in keywords:

        if (keyword in dictionary):

            index = dictionary[keyword]

            if (index in training_sample_counts):

                count = training_sample_counts[index]

                keyword_indexes_list.append((dictionary[keyword], keyword, count))

    

    

    print('mean count of the words in the training set: ' + str(mean(all_counts)))

    print('count of the keywords in training set')

    print(keyword_indexes_list)

    

    

    return word_target, word_context, labels



#custom made skipgram training data for specifically training the embeddings for the

#contexts of the keywords specified

def get_skip_grams_keywords(data_skipgram, vocab_size, keywords, dictionary, reversed_dictionary):

    num_training_samples = 200000

    window_size = 3

    

    #getting the keyword indexes

    keyword_indexes = []

    for keyword in keywords:

        keyword_indexes.append(dictionary[keyword])

    

    #getting all the list of nearby indexes from the data

    word_contexts = {}

    for keyword_index in keyword_indexes:

        word_contexts[keyword_index] = {}

        word_contexts[keyword_index]['context_set'] = set()

        word_contexts[keyword_index]['null_set'] = set()

    for index in range(len(data_skipgram)):

        if (reversed_dictionary[data_skipgram[index]] in keywords):

            min_index = max(0, index-window_size)

            max_index = min(len(data_skipgram), index+window_size)

            context_words = []

            for context_index in range(min_index, max_index):

                if (context_index != index) and (context_index != 0):

                    context_words.append(data_skipgram[context_index])

            for context_word in context_words:

                word_contexts[data_skipgram[index]]['context_set'].add(context_word)

    

    #getting the set of words not in the window of the keywords

    for key, value in word_contexts.items():

        for _, pot_index in  dictionary.items():

            if (pot_index != 0) and (pot_index != key) and (pot_index not in value['context_set']):

                word_contexts[key]['null_set'].add(pot_index)

            

    #randomly producing the training set

    start_time = datetime.now()

    word_target = np.ndarray(shape = (num_training_samples, ), dtype=np.int32)

    word_context = np.ndarray(shape = (num_training_samples, ),  dtype=np.int32)

    labels = list()

    for index in range(num_training_samples):

        target_index = random.choice(keyword_indexes)

        word_target[index] = target_index

        label_index = random.randint(0, 1)

        labels.append(label_index)

        if (label_index):

            word_context[index] = random.sample(word_contexts[target_index]['context_set'], 1)[0]

        else:

            word_context[index] = random.sample(word_contexts[target_index]['null_set'], 1)[0]

    end_time = datetime.now()

    time_lapsed = end_time - start_time

    print('generating the training data took: {}s'.format(time_lapsed.seconds))

    

    return word_target, word_context, labels

        

    

    

    

            

    

    

                

                

        

    
def cosine_distance(inputs):

    # cosine distance function (retusn a layer of shape (1, ))

    x, y = inputs

    x = K.l2_normalize(x, axis=1)

    y = K.l2_normalize(y, axis=1)

    similarity = -K.mean(x * y, axis=1, keepdims=False)

    return similarity



def cos_dist_output_shape(shapes):

    return (1,)







def dot_product(inputs):

    # dot product (retusn a layer of shape (1, ))

    x, y = inputs

    out_sum = K.sum(x * y,axis=1, keepdims=False)

    return out_sum



def dot_product_output_shape(shapes):

    return (1,)
def get_models(vocab_size):

    #thi function gets the training model and the validation model

    vector_dim = 128

    # inputs to the network will be single word and context word

    input_target = Input((1,))

    input_context = Input((1,))



    #embedding layer

    embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

    target = embedding(input_target)

    target = Reshape((vector_dim, 1))(target)

    context = embedding(input_context)

    context = Reshape((vector_dim, 1))(context)

    

    #dot product layer

    dot_prod = Lambda(dot_product, output_shape=dot_product_output_shape)([target, context])

    dot_prod = Reshape((1,))(dot_prod)

    # add the sigmoid output layer

    output = Dense(1, activation='sigmoid')(dot_prod)

    

    # create the primary training model

    model = Model(input=[input_target, input_context], output=output)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')



    

    # create a validation model for getting the similar matrixes

    similarity = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([target, context])

    validation_model = Model(input=[input_target, input_context], output=similarity)

    

    return model, validation_model



    
def setup_gpu():

    # this function mainly sets up the gpu for the use for the model training

    # it returns the device name for the gpu that will be used 

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    #tf.debugging.set_log_device_placement(True)



    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:

      # Restrict TensorFlow to only use the first GPU

      try:

        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

      except RuntimeError as e:

        # Visible devices must be set before GPUs have been initialized

        print(e)



    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:

      try:

        # Currently, memory growth needs to be the same across GPUs

        for gpu in gpus:

          tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

      except RuntimeError as e:

        # Memory growth must be set before GPUs have been initialized

        print(e)



    device_name = tf.test.gpu_device_name()

    if "GPU" not in device_name:

        print("GPU device not found")

    print('Found GPU at: {}'.format(device_name))

    

    return device_name

    
def train_model(model, device_name, word_target, word_context, labels):

    # this function basically trains the keras model

    # this is currently set to 200000 for completing in reasonable time. For better accuracy the recommendation is 2000000

    epochs = 200000

    

    word_target_input = np.zeros((1,))

    word_context_input = np.zeros((1,))

    labels_input = np.zeros((1,))

    start_time = datetime.now()

    loss_function = []

    with tf.device(device_name):        

        for cnt in range(epochs):

            idx = np.random.randint(0, len(labels)-1)

            word_target_input[0,] = word_target[idx]

            word_context_input[0,] = word_context[idx]

            labels_input[0,] = labels[idx]

            loss = model.train_on_batch([word_target_input, word_context_input], labels_input)

            loss_function.append(loss)

            if cnt % 1000 == 0 and cnt > 0:

                time_lapse = datetime.now() - start_time

                print("Iteration {}, loss={} time passed: {}s, expected time of completion: {}s and is {}% complete".

                      format(cnt, loss, time_lapse.seconds, epochs/cnt*time_lapse.seconds, (cnt*1.0/epochs)*100))

                      

    fig, ax = plt.subplots()

    ax.plot(range(len(loss_function)), loss_function)



    ax.set(xlabel='index', ylabel='total loss',

           title='Loss function of the NN training for the embedding model')

    ax.grid()

    plt.show()



    return model            

            

    
def get_sim(valid_word_idx, vocab_size, validation_model):

    # this function gets the list of 

    sim = np.zeros((vocab_size,))

    in_arr1 = np.zeros((1,))

    in_arr2 = np.zeros((1,))

    for i in range(vocab_size):

        in_arr1[0,] = valid_word_idx

        in_arr2[0,] = i

        out = validation_model.predict_on_batch([in_arr1, in_arr2])

        sim[i] = out

    return sim



def run_sim(keyword, top_k, dictionary,reversed_dictionary, validation_model):

    if (keyword not in dictionary):

        return

    valid_word_idx = dictionary[keyword]

    sim = get_sim(valid_word_idx, len(dictionary), validation_model)

    nearest = (-sim).argsort()[1:top_k + 1]

    keywords = []

    for k in range(top_k):

        close_word = reversed_dictionary[nearest[k]]

        keywords.append(close_word)

    return keywords





def update_keyword_questions(keywords_questions, keyword, seconday_keywords):

    for item in keywords_questions.items():

        if (keyword in item[1]['keywords']):

            weight = 0.5*item[1]['weights'][item[1]['keywords'].index(keyword)]

            for sec_keyword in seconday_keywords:

                item[1]['keywords'].append(sec_keyword)

                item[1]['weights'].append(weight)

    return keywords_questions

            

            



def get_top_k_similar_keywords(keywords, weights, keywords_questions, top_k, dictionary,reversed_dictionary, validation_model):

    new_keywords = []

    new_weights = []

    num_done = 0

    start_time = datetime.now()

    for keyword, weight in  zip(keywords, weights):

        seconday_keywords = run_sim(keyword, top_k, dictionary, reversed_dictionary, validation_model)

        if not seconday_keywords:

            continue

        print('got nearest keywords  {} for keyword {}: '.format(seconday_keywords, keyword) )

        keywords_questions = update_keyword_questions(keywords_questions, keyword, seconday_keywords)

        for sec_keyword in seconday_keywords:

            if sec_keyword not in keywords:

                new_keywords.append(sec_keyword)

                new_weights.append(weight*0.5)

        num_done += 1

        time_lapse = datetime.now() - start_time

        print('Done: {} out of {}, completed {}% time passed {}s and estimated time to complete {}s'.format(num_done, 

                                                                                                            len(keywords),

                                                                                                            num_done*1.0/len(keywords)*100,

                                                                                                            time_lapse.seconds,

                                                                                                            len(keywords)/num_done*time_lapse.seconds))

    

    

    keywords.extend(new_keywords)

    weights.extend(new_weights)

    

    print('final list of keywords and weights')

    print(keywords, weights)

    print('updated keywords within the questions')

    print(keywords_questions)

    return keywords, weights, keywords_questions

                

        

        

        
def expand_keywords(keywords, weights,keywords_questions, data):



    start_time = datetime.now()

    word_list = get_list_of_words(data)

    time_lapse = datetime.now() - start_time

    print('getting all the words in the dataset took: ' + str(time_lapse.seconds) + 's ' )



    start_time = datetime.now()

    dictionary_size = 10000

    data_skipgram, dictionary, reversed_dictionary, vocab_size = build_dataset(word_list, dictionary_size, keywords)

    time_lapse = datetime.now() - start_time

    print('getting the most common words and their counts from data took: ' + str(time_lapse.seconds) + 's ' )



    print('getting skipgrams')

    #word_target, word_context, labels = get_skip_grams(data_skipgram, vocab_size, keywords, dictionary)

    word_target, word_context, labels = get_skip_grams_keywords(data_skipgram, vocab_size, keywords, dictionary, reversed_dictionary)

    print('got skipgrams')

    print('setting up the models')

    model, validation_model = get_models(vocab_size)

    print('got the models')





    print('setting up the gpu')

    device_name = setup_gpu()

    print('gpu is setup')

    print('training the model')

    model = train_model(model, device_name, word_target, word_context, labels)

    print('model is trained')

    top_k = 5

    keywords, weights, keywords_questions = get_top_k_similar_keywords(keywords, weights, keywords_questions, top_k, dictionary,reversed_dictionary, validation_model)

    return keywords, weights, keywords_questions
def get_score(text, keywords, weights):

    # get the score of a given text, given the weights and keywords

    tokens = clean_up_and_tokenize_text(text)

    score = 0

    for token in tokens:

        if token in keywords:

            score += weights[keywords.index(token)]

    return score



def score_paper(json_doc, keywords, weights):

    #json_doc is the dictionary object representing the json

    #this function basicaly goes through the titel, abstract and text and

    #does a weighted sum, whenever a keyword is hit

    

    score = 0

    

    has_title = True

    has_text = True

    has_abstract = True

    if 'paper_id' in json_doc:

        paper_id = json_doc['paper_id']

    else:

        print('ERROR paper id was not present')

        return

    if 'metadata' in json_doc:

        if 'title' in json_doc['metadata']:

            title = json_doc['metadata']['title']

        else:

            has_title = False

            print('ERROR no title for paper id: ' + str(paper_id))

    else:

        has_title = False

        print('ERROR no metadata for paper id: ' + str(paper_id))

    

    if 'abstract' in json_doc:

        abstract = json_doc['abstract']

    else:

        has_abstract = False

        #print('ERROR no abstract for paper id: ' + str(paper_id))

        

    if 'body_text' in json_doc:

        body_tect = json_doc['body_text']

    else:

        has_text = False

        #print('ERROR no text for  paper id: ' + str(paper_id))

    

    #going through the title

    if (has_title):

        score = get_score(title, keywords, weights)

                

    #going through the abstract

    if (has_abstract):

        for paragraph in abstract:

            paragraph_text = paragraph['text']

            paragraph_text = paragraph_text.replace('et al.', '')

            score = get_score(paragraph_text, keywords, weights)

            

    #going through the body

    if (has_text):

        sent_number = 0

        paragraph_number = 0

        for paragraph in body_tect:

            paragraph_text = paragraph['text']

            paragraph_text = paragraph_text.replace('et al.', '')

            score = get_score(paragraph_text, keywords, weights)

    

    return paper_id, score





def get_list_of_top_k_papers_thread(input_dict):

    # the main function for getting the top k papers from the dataset

    # given the keywords and weights

    keywords = input_dict['keywords']

    weights = input_dict['weights']

    top_k = input_dict['topk']

    thread_num = input_dict['thread']

    num_files = 0

    path = input_dict['path']

    for dirname, _, filenames in os.walk(path):

        for filename in filenames:

            num_files = num_files + 1

    print('Total number of files = ' + str(num_files))

    total_count = 0

    start_time = datetime.now()



    score_data = [];

    



    for dirname, _, filenames in os.walk(path):

        for filename in filenames:

            if (filename.endswith('.json')):

                with open(os.path.join(dirname, filename), 'r') as f:

                    sample_json_dict = json.load(f)

                    paper_id, score = score_paper(sample_json_dict, keywords, weights)

                    if (score):

                        score_data.append([paper_id, score])

                    total_count = total_count + 1

                    if (total_count%500 == 0):

                        time_now = datetime.now()

                        elapsed_time = time_now - start_time

                        print('Processed: ' + str(total_count) + ' for thread' + str(thread_num) + ' Documents remaining: ' 

                              + str(num_files - total_count) + ' done: ' 

                              + str((total_count*1.0)/(num_files)*100.0) + '%'

                                    + ' expected time: ' + str(num_files/total_count*elapsed_time.seconds)

                                    + 's'+ ' time: ' + str(elapsed_time.seconds) + 's')

    return score_data;

                        

                        

def get_list_of_top_k_papers(keywords, weights, top_k):

    root_path = '/kaggle/input'

    multiprocess_dic = [{ 'path' : root_path + '/CORD-19-research-challenge/biorxiv_medrxiv/',

                          'keywords' : keywords,

                          'weights' : weights,

                          'topk' : top_k,

                          'thread' : 0

                        },

                        { 'path' : root_path + '/CORD-19-research-challenge/comm_use_subset/',

                          'keywords' : keywords,

                          'weights' : weights,

                          'topk' : top_k,

                          'thread' : 1

                        },

                        { 'path' : root_path + '/CORD-19-research-challenge/custom_license/',

                          'keywords' : keywords,

                          'weights' : weights,

                          'topk' : top_k,

                          'thread' : 2

                        },

                        { 'path' : root_path + '/CORD-19-research-challenge/noncomm_use_subset/',

                          'keywords' : keywords,

                          'weights' : weights,

                          'topk' : top_k,

                          'thread' : 3

                        }]



    pool = mp.Pool(processes = mp.cpu_count())

    result = pool.map(get_list_of_top_k_papers_thread, multiprocess_dic)

    score_data = np.concatenate(result)

    columns = ['paper_id', 'score']

    

    del result

    

    score_dataframe = pd.DataFrame(score_data, columns=columns)    

    score_dataframe.dropna(inplace=True)

    score_dataframe = score_dataframe.astype({'score': np.float64})

    score_dataframe.sort_values(by=['score'], ascending=False, inplace=True)

    

    x = range(len(score_dataframe))

    y = score_dataframe['score'].tolist()



    fig, ax = plt.subplots()

    ax.plot(x, y)



    ax.set(xlabel='index', ylabel='score of the paper',

           title='Scores of the papers based on the keywords match')

    ax.grid()

    plt.show()

    

    top_size = min(top_k, len(score_dataframe))

    return score_dataframe['paper_id'].tolist()[:top_size]

def get_unique_keywords(keywords_questions):

    # gets the unique set of keywords as a list and their corresponding weights initialized as ones

    keywords  = []

    for item in keywords_questions.items():

        keywords.extend(item[1]['keywords'])

    keywords_set = set(keywords)

    return list(keywords_set), [1]*len(keywords_set)
def get_top_k_scoring_papers(keywords, weights, data, top_n):

    # given the dataframe that has been cleaned and pre-processed

    # this function scores and gets the top N highest scoring paper ids

    paper_id = ''

    paper_score = 0

    paper_score_data = []

    for _, row in data.iterrows():

        if row['paper_id'] != paper_id:

            paper_score_data.append([paper_id, paper_score])

            paper_score = 0

            paper_id = row['paper_id']

        paper_score += get_score(row['content'], keywords, weights)

        paper_id = row['paper_id']

    #for the last paper

    paper_score_data.append([paper_id, paper_score])

    columns = ['paper_id', 'score']

    score_dataframe = pd.DataFrame(paper_score_data, columns=columns)    

    score_dataframe.dropna(inplace=True)

    score_dataframe = score_dataframe.astype({'score': np.int64})

    score_dataframe.sort_values(by=['score'], ascending=False, inplace=True)

    

    x = range(len(score_dataframe))

    y = score_dataframe['score'].tolist()



    fig, ax = plt.subplots()

    ax.plot(x, y)



    ax.set(xlabel='index', ylabel='score of the paper',

           title='Scores of the final papers based on the final list of keywords match')

    ax.grid()

    plt.show()

    

    return score_dataframe['paper_id'].tolist()[:top_n]



def get_text_from_json(json_doc, paper_ids):

    # gets the text from the json document

    # concatenates all the paragraphs togather

    # text_type can either be a 'title' or 'obstract'

    paper_data = {}

    if 'paper_id' in json_doc:

        paper_id = json_doc['paper_id']

        if (paper_id not in paper_ids):

            return

        paper_data['paper_id'] = paper_id

    else:

        print('ERROR paper id was not present')

        

    paper_data['paper_data'] = {}

    if 'metadata' in json_doc:

        if 'title' in json_doc['metadata']:

            paper_data['paper_data']['title'] =  json_doc['metadata']['title']

        else:

            has_title = False

            print('ERROR no title for paper id: ' + str(paper_id))

            

    else:

        has_title = False

        print('ERROR no metadata for paper id: ' + str(paper_id))

        

    if 'abstract' in json_doc:

        abstract = json_doc['abstract']

        abstract_text = ''

        for paragraph in abstract:

            paragraph_text = paragraph['text']

            abstract_text += paragraph_text + ' '

        paper_data['paper_data']['abstract'] = abstract_text

    else:

        print('ERROR no abstract for paper id: ' + str(paper_id))

        

    return paper_data

    





def get_text_for_papers(paper_ids):

    # iterates through the data set and gets the text for the specific set of papers

    # main function for getting the final report

   

    num_files = 0

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            num_files = num_files + 1

    print('Total number of files = ' + str(num_files))

    total_count = 0

    start_time = datetime.now()



    text_data = [];

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            if (filename.endswith('.json')):

                with open(os.path.join(dirname, filename), 'r') as f:

                    sample_json_dict = json.load(f)

                    paper_data = get_text_from_json(sample_json_dict, paper_ids)

                    if paper_data:

                        text_data.append(paper_data)

                    total_count = total_count + 1

                    if (total_count%1000 == 0):

                        time_now = datetime.now()

                        elapsed_time = time_now - start_time

                        print('Processed: ' + str(total_count) + ' Documents remaining: ' 

                              + str(num_files - total_count) + ' done: ' 

                              + str((total_count*1.0)/(num_files)*100.0) + '%'

                                    + ' expected tiem: ' + str(num_files/total_count*elapsed_time.seconds)

                                    + 's'+ ' time: ' + str(elapsed_time.seconds) + 's')

    return text_data

        

        

        

    
keywords_questions = {

    'q1': {

        'keywords': ['drug', 'naproxen', 'clarithromycin', 'minocycline', 'viral', 'replication', 'vaccine', 'antibody', 'complication'],

        'weights' : [1, 3, 3, 3, 1, 1, 1, 2, 2]

    },

    'q2': {

        'keywords': ['animal', 'rat', 'monkey'],

        'weights' : [2, 2, 2, 2]

    },

    'q3': {

        'keywords': ['therapeutic', 'antiviral', 'agents'],

        'weights' : [1, 1, 1]

    },

    'q4': {

        'keywords': ['alternative', 'therapeutic', 'production', 'capacity', 'distribution'],

        'weights' : [3, 2, 2, 1, 1, 1]

    },

    'q5': {

        'keywords': ['universal', 'vaccine'],

        'weights' : [3, 3]

    },

    'q6': {

        'keywords': ['animal', 'challenge'],

        'weights' : [3, 3]

    },

    'q7': {

        'keywords': ['prophylaxis', 'clinical'],

        'weights' : [3, 1]

    },

    'q8': {

        'keywords': ['enhanced', 'disease', 'vaccination', 'vaccine'],

        'weights' : [3, 3, 1, 1]

    },

    'q9': {

        'keywords': ['immune', 'response', 'vaccine','animal',  'therapeutic'],

        'weights' : [3, 3, 1, 2, 1, 1]

    },

    'q10': {

        'keywords': ['ade', 'antibody', 'enhancement', 'recipient', 'complication'],

        'weights' : [3, 3, 3, 2, 3]

    }

}

#declare the global list of stop words

stop_words = get_all_stop_words(nlp)

keywords, weights = get_unique_keywords(keywords_questions)

#because we want to finish the execution in a reasonable time the number of iterations is set to 1

#for a more comprehencive list of keywords we can have this set to 2 (which will add about 1.5 hour to the computation times, the number of keywords will multiply by 5)

num_iterations = 1

top_k = 500

for iteration in range(num_iterations):

    paper_ids = get_list_of_top_k_papers(keywords, weights, top_k)

    best_ppaper_data = get_dataframe_from_paper_ids(paper_ids)

    keywords, weights, keywords_questions = expand_keywords(keywords, weights, keywords_questions, best_ppaper_data)

    del best_ppaper_data

    gc.collect()

    

#getting the paper ids and the best data one last time

paper_ids = get_list_of_top_k_papers(keywords, weights, top_k)

best_ppaper_data = get_dataframe_from_paper_ids(paper_ids)

# now that we have all he papers that we need, we will print out the reports for all the categories

report_cat = ['Effectiveness of drugs being developed and tried to treat COVID-19 patients.',

              'Exploration of use of best animal models and their predictive value for a human vaccine.',

              'Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.',

              'Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.',

              'Efforts targeted at a universal coronavirus vaccine.',

              'Efforts to develop animal models and standardize challenge studies',

              'Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers',

              'Approaches to evaluate risk for enhanced disease after vaccination',

              'Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]',

              'Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.'

             ]

top_n = 10

paper_info = {}

for index in range(len(report_cat)):

    print('category: ' + str(report_cat[index]))

    question_number = 'q'+str(index+1)            

    keywords = keywords_questions[question_number]['keywords']

    weights = keywords_questions[question_number]['weights']

    paper_ids = []

    paper_ids = get_top_k_scoring_papers(keywords, weights, best_ppaper_data, top_n)

    paper_info['q'+str(index+1)] = {}

    paper_info['q'+str(index+1)]['paper_ids'] = paper_ids 

    paper_info['q'+str(index+1)]['category_topic'] = report_cat[index]

    

del best_ppaper_data

#getting the unique list of paper ids

unique_paper_ids = set()

for value in paper_info.values():

    for paper_id in value['paper_ids']:

        unique_paper_ids.add(paper_id)

    



final_paper_data = get_text_for_papers(list(unique_paper_ids))



#now we're fetching the paper title and abstract for each of the paper ids

for index in range(len(report_cat)):

    top_paper_data = []

    for paper_id in paper_info['q'+str(index+1)]['paper_ids']:

        for p_data in final_paper_data:

            if (p_data['paper_id'] == paper_id):

                top_paper_data.append(p_data)

                break

    paper_info['q'+str(index+1)]['top_paper_data'] = top_paper_data

    
import pprint

pp = pprint.PrettyPrinter(indent = 1,width = 300)

pp.pprint('The following are the list of top 10 paper ids for each research category')

for index in range(len(report_cat)):

    pp.pprint('')

    pp.pprint('')

    pp.pprint('Paper Ids for Category:')

    pp.pprint('******************************************')

    pp.pprint(paper_info['q'+str(index+1)]['category_topic'])  

    pp.pprint('******************************************')

    pp.pprint(paper_info['q'+str(index+1)]['paper_ids'])

              

pp.pprint('')

pp.pprint('')

pp.pprint('')

pp.pprint('')

pp.pprint('')



pp.pprint('The following are the titles of top 10 titles for each research category')

for index in range(len(report_cat)):

    pp.pprint('')

    pp.pprint('')

    pp.pprint('Titles for Category:')

    pp.pprint('-----------------------------------------')

    pp.pprint(paper_info['q'+str(index+1)]['category_topic'])  

    pp.pprint('-----------------------------------------')

    for top_paper in paper_info['q'+str(index+1)]['top_paper_data']:

        pp.pprint(top_paper['paper_data']['title'])



pp.pprint('')

pp.pprint('')

pp.pprint('')

pp.pprint('')

pp.pprint('')

pp.pprint('The following are the abstracts of top 10 abstracts for each research category')

for index in range(len(report_cat)):

    pp.pprint('')

    pp.pprint('')

    pp.pprint('Abstracts for Category:')

    pp.pprint('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    pp.pprint(paper_info['q'+str(index+1)]['category_topic'])  

    pp.pprint('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    for top_paper in paper_info['q'+str(index+1)]['top_paper_data']:

        if ('abstract' in top_paper['paper_data']):

            pp.pprint('')

            pp.pprint(top_paper['paper_data']['abstract'])

                   

pp.pprint('THE END')