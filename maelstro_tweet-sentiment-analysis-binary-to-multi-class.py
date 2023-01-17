# Listing the directories with datasets

import os



dir_list = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        dir_list.append(os.path.join(dirname, filename))

        

print(dir_list)
#Importing dependencies/packages



# Utility dependencies

import re

import pickle



# Graphic/math dependencies

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud



# Algorithmic dependencies - used to classify the text

from sklearn.cluster import KMeans, DBSCAN

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer

import nltk

from nltk.corpus import stopwords
# Constants - used in the notebook

ENCODING = "ISO-8859-1"

COLUMN_HEADERS = ["sentiment", "ids", "date", "flag", "user", "text"]



# Data exploration - importing the dataset and value lookup

data = pd.read_csv(dir_list[1],

                  encoding=ENCODING,

                  names=COLUMN_HEADERS)
# Data lookup

print(data.head())



# Show the example tweet

data['text'][0]
# Extract the 'text' column from the dataset

texts = data['text']



# Defining dictionary containing all emojis with their meanings.

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 

          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',

          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 

          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',

          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',

          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 

          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}



## Defining set containing all stopwords in english.

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',

             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',

             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',

             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 

             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',

             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',

             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',

             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',

             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',

             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',

             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',

             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 

             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',

             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',

             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",

             "youve", 'your', 'yours', 'yourself', 'yourselves']



stopword_nltk = stopwords.words('english')



def preprocess(textdata):

    processedText = []

    

    # Create Lemmatizer and Stemmer.

    wordLemm = WordNetLemmatizer()

    

    # Defining regex patterns.

    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"

    userPattern       = '@[^\s]+'

    alphaPattern      = "[^a-zA-Z0-9]"

    sequencePattern   = r"(.)\1\1+"

    seqReplacePattern = r"\1\1"

    

    for tweet in textdata:

        tweet = tweet.lower()

        

        # Replace all URls with 'URL'

        tweet = re.sub(urlPattern,' URL',tweet)

        # Replace all emojis.

        for emoji in emojis.keys():

            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        

        # Replace @USERNAME to 'USER'.

        tweet = re.sub(userPattern,' USER', tweet)        

        # Replace all non alphabets.

        tweet = re.sub(alphaPattern, " ", tweet)

        # Replace 3 or more consecutive letters by 2 letter.

        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)



        tweetwords = ''

        for word in tweet.split():

            # Checking if the word is a stopword.

            if word not in stopword_nltk:

                if len(word)>1:

                    # Lemmatizing the word.

                    word = wordLemm.lemmatize(word)

                    tweetwords += (word+' ')

            

        processedText.append(tweetwords)

        

    return processedText
# Preprocessing the texts

import time

start = time.time()

processed_text = preprocess(texts)

print('Text processing complete.')

print(f'It took {round(time.time() - start)} seconds to complete this task.')
# Select some random tweets

example_tweets = processed_text[1000:2000]



plt.figure(figsize=(16,16))

wc = WordCloud(max_words=100, width=1200, height=600,

              collocations=False).generate(" ".join(example_tweets))

plt.imshow(wc)
# Vectorizer declaration and data transformation

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=800000)

start = time.time()

vectorizer.fit(processed_text)

print(f'Vectorizer fitting ended in {round(time.time()-start)} seconds')

print(f'Number of feature_words: {len(vectorizer.get_feature_names())}')
# Dataset transformation

start = time.time()

transformed_tweets = vectorizer.transform(processed_text)

print(f'Data transformation ended in {round(time.time()-start)} seconds')
# Train-test split

text_train, text_test = train_test_split(transformed_tweets, test_size=0.2, random_state=42)
# Function to present training results

def show_training_results(model, num_clusters):

    order_centroids = model.cluster_centers_.argsort()[:,::-1]

    terms = vectorizer.get_feature_names()

    

    for i in range(num_clusters):

        print("Cluster %d:" % i)

        for ind in order_centroids[i, :100]:

            print(' %s' % terms[ind])

            

# Function to test the model

def test_prediction(model, tweet_number):

    print("Prediction")

    try:

        to_predict = vectorizer.transform([data['text'][tweet_number]])

    except:

        print("Exception, wrong number")

        to_predict = vectorizer.transform([data['text'][0]])

    finally:

        prediction = model.predict(to_predict)

        print(prediction)

        

def training_accuracy(model):

    result = 0

    predictions = model.predict(text_test)

    for i in range(text_test.shape[0]):

        if data['sentiment'][i] // 4 == predictions[i]:

            result +=1

    return result
# KMeans

start = time.time()

km = KMeans(n_clusters=20, init='random', n_init=1, max_iter=100, tol=1e-5, random_state=2200)

km.fit(text_train)

print(f'Model training finished in {round(time.time() - start)} seconds.')



show_training_results(km, 20)
# KMeans training accuracy

acc = training_accuracy(km)

print(f'KMeans accuracy: {acc / text_test.shape[0] * 100} percent.')
# KMeans++

start = time.time()

kmpp = KMeans(n_clusters=20, init='k-means++', n_init=1, max_iter=10000, tol=1e-5, random_state=1)

kmpp.fit(transformed_tweets)

print(f'KMeans++ model training finished in {round(time.time() - start)} seconds.')



show_training_results(kmpp, 20)
# KMeans training accuracy

kmpp_acc = training_accuracy(kmpp)

print(f'KMeans++ accuracy: {kmpp_acc / text_test.shape[0] * 100} percent.')
# # DBSCAN

# start = time.time()

# mdl_dbscan = DBSCAN(eps=100.0)

# mdl_dbscan.fit(text_train)

# print(f'DBSCAN model training finished in {round(time.time() - start)} seconds.')



# show_training_results(mdl_dbscan)



# DBSCAN training accuracy

# dbscan_acc = training_accuracy(mdl_dbscan)

# print(f'DBSCAN accuracy: {dbscan_acc / text_test.shape[0] * 100} percent.')
# Packing the models



with open('./vectorizer-ngram(1,2).pickle', 'wb') as file:

    pickle.dump(vectorizer, file)



with open('./kmeans-model.pickle', 'wb') as file:

    pickle.dump(km, file)



with open('./kmeans-pp-model.pickle', 'wb') as file:

    pickle.dump(kmpp, file)
# Load the file into a DataFrame

word_base = pd.read_excel('/kaggle/input/characterspecificwordbase/character_phrases.xlsx', header=0, sheet_name=0, index_col=None)



# Set description

word_base
# Text preprocessing

word_base_dict = {}

for i in range(word_base.shape[0]):

    word_base_dict[word_base.TYPE[i]] = word_base.PHRASES[i].split(",")

    

lemmatizer = WordNetLemmatizer()

def preprocess_base(dataset: dict):

    

    # Word lemmatizer

    processed_text = {}

    for key, phrases in dataset.items():

        new_phrases = []

        for phrase in phrases:

            # Convert all letters to lowercase

            phrase = phrase.lower()

            

            # Remove all non-alphanumeric characters

            new_phrases.append(re.sub("[^a-zA-Z0-9/]", " ", phrase))

        processed_text[key] = new_phrases    

    return processed_text



processed_wb_dict = preprocess_base(word_base_dict)



words = []              # Tokenized_words

key_phrase_sets = []    # Character-phrase tuples

classes = []            # Character types

# Word tokenization

for key, phrases in processed_wb_dict.items():

    for phrase in phrases:

        w = nltk.word_tokenize(phrase)

        words.extend(w)

        key_phrase_sets.append((w, key))

        

        if key not in classes:

            classes.append(key)
# Training data preparation

training_set = []

output_empty = [0] * len(classes)



for text_pair in key_phrase_sets:

    # Bag initialization

    bag = []

    

    # List of tokenized words

    pattern_words = text_pair[0]

    

    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    

    # Fill the bag of words with '1' if the word match is found in the current pattern

    for w in words:

        bag.append(1) if w in pattern_words else bag.append(0)

        

    output_row = list(output_empty)

    output_row[classes.index(text_pair[1])] = 1

    

    training_set.append([bag, output_row])

    

# Shuffle the features into np.array

import random



random.shuffle(training_set)

training_set = np.array(training_set)

X_set = training_set[:, 0]

y_set = training_set[:, 1]



X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.25, 

                                                    random_state=1523)
# Model definition



# Keras dependencies

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten, LSTM

from keras.layers import GlobalMaxPooling1D

from keras.models import Model

from keras.layers import Input

from keras.optimizers import SGD

from keras.preprocessing.sequence import pad_sequences



# Padding the training/testing dataset

maxlen = 0

for i in range(len(X_train)):

    maxlen = len(X_train[i]) if len(X_train[i]) > maxlen else maxlen



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



y_train = np.array([np.array(row) for row in y_train])

y_test = np.array([np.array(row) for row in y_test])
# Feedforward neural network

ff_model = Sequential()

ff_model.add(Dense(256, input_shape=(len(X_train[0]),), activation='relu'))

ff_model.add(Dropout(0.5))

ff_model.add(Dense(128, activation='relu'))

ff_model.add(Dropout(0.5))

ff_model.add(Dense(len(y_train[0]), activation='relu'))



sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

ff_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Training the feedforward model

ff_hist = ff_model.fit(np.array(X_train), np.array(y_train), epochs=50, batch_size=6, verbose=1,

                   validation_split=0.1)

# list all data in history

print(ff_hist.history.keys())





plt.plot(ff_hist.history['accuracy'])

plt.plot(ff_hist.history['val_accuracy'])



plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train','Test'], loc='upper left')

plt.show()



plt.plot(ff_hist.history['loss'])

plt.plot(ff_hist.history['val_loss'])



plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train','Test'], loc='upper left')

plt.show()
# Convolutional neural network - model test



def clean_up_sentence(sentence):

    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    return sentence_words



def bow(sentence, words, show_details=True):

    # tokenize the pattern

    sentence_words = clean_up_sentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix

    bag = [0]*len(words)

    for s in sentence_words:

        for i,w in enumerate(words):

            if w == s:

                # assign 1 if current word is in the vocabulary position

                bag[i] = 1

                if show_details:

                    print ("found in bag: %s" % w)

    return(np.array(bag))



def predict_class(sentence, model):

    # filter out predictions below a threshold

    p = bow(sentence, words,show_details=False)

    res = model.predict(np.array([p]))[0]

    ERROR_THRESHOLD = 0.25

    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    # sort by strength of probability

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:

        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list



predict_class(processed_text[10000], ff_model)