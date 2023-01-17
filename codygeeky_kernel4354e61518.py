import pandas as pd

import numpy as np

%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

from PIL import Image

import chakin

import re

import os

import string

import nltk

from nltk.tokenize import WordPunctTokenizer

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from textblob import TextBlob

from sklearn.model_selection  import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors



from keras.models import Sequential

from keras.layers import Dense, Flatten, Embedding, LSTM, SpatialDropout1D, Input, Bidirectional,Dropout

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
airline_tweets = pd.read_csv('../input/Tweets.csv')
username = '@[A-Za-z0-9]+'

url = 'https?://[^ ]+'

link = 'www.[^ ]+'

combined_p = '|'.join((username, url, link))



##  Cleaning Function 

def tweet_cleaner( tweet ):

    #

    # 1. Remove non-informative text    

    tweet = re.sub(combined_p, '', tweet)

    #

    # 2. Remove non-letters

    tweet = re.sub("[^a-zA-Z]"," ", tweet)

    #

    # 3. Convert words to lower case

    lower_tweet = tweet.lower()

    #

    # 4. Tokenize tweet 

    tok = WordPunctTokenizer()

    tweet_words = [x for x in tok.tokenize(lower_tweet) if len(x) > 1]

    #

    # 5. remove punctuation from each word

    table = str.maketrans('','', string.punctuation)

    stripped = [w.translate(table) for w in tweet_words]

    #

    # 6. remove stop words

    stops = set(stopwords.words("english"))

    words = [w for w in stripped if not w in stops]

    #

    # 7. combine words and return cleaned tweet

    return (" ".join(words)).strip()
normalized_tweets = []

for tweet in airline_tweets.text:

    normalized_tweets.append(tweet_cleaner(tweet))

    
"""

Modify plot size

"""

plot_size = plt.rcParams["figure.figsize"]

plot_size[0] =8

plot_size[1] = 8

plt.rcParams["figure.figsize"] = plot_size

sentiment = airline_tweets.airline_sentiment.value_counts().to_frame()

sentiment.columns = ['count']

print(sentiment)

# print(airline_tweets.airline_sentiment.value_counts().to_frame())

airline_tweets.airline_sentiment.value_counts().plot(kind='pie',autopct='%1.0f%%')

plt.title('Fig 1. Distribution of tweets sentiments', fontsize=18)
airline_sentiment = airline_tweets[airline_tweets.airline_sentiment == 'negative'].groupby(['airline']).airline_sentiment.count()

p = pd.DataFrame(airline_sentiment.sort_values(ascending=False))

p.columns = ['Negative Count']

print (p)

airline_sentiment = airline_sentiment.plot(kind='bar', color=sns.color_palette('hls'))

airline_sentiment.set_xlabel('Airline')

airline_sentiment.set_ylabel('Negative tweets count')

plt.title('Fig 2. Negative Tweets per Airline', fontsize=18)
airline_sentiment = (airline_tweets.groupby(['airline', 'airline_sentiment']).size()/ airline_tweets.groupby('airline').size()*100).unstack()

airline_sentiment=airline_sentiment.plot(kind='bar')

airline_sentiment.set_xlabel('Airline')

airline_sentiment.set_ylabel('Percentage (100%)')

plt.title('Fig 3. Distribution of sentiment per Airline', fontsize=18)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
airline_tweets.negativereason.value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.title('Fig 4. Negative Reasons Percentages', fontsize=18)

plt.ylabel('Negative Reasons')
#Visualize negaive reasons per airline

pd.crosstab(airline_tweets.airline, airline_tweets.negativereason).apply(lambda x: x / x.sum() * 100, axis=1).plot(kind='bar',stacked=True)

plt.title('Fig 5. The Reasons Customers React Negatively to Each Airline in Frequency', fontsize=18)

plt.xlabel('Airline')

plt.ylabel('Percentage of Reason')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
top_negative_reasons= airline_tweets.negativereason[airline_tweets.negativereason.isin(["Customer Service Issue", "Late Flight", "Can't Tell", "Cancelled Flight" ])]

airline_tweets['top_negative_reasons']=top_negative_reasons

airline_sentiment = (airline_tweets.groupby(['airline', 'top_negative_reasons']).size()/ airline_tweets.groupby('airline').size()*100).unstack()

airline_sentiment=airline_sentiment.plot(kind='bar')

airline_sentiment.set_xlabel('Airline')

airline_sentiment.set_ylabel('Percentage (100%)')

plt.title('Fig 3. Distribution of sentiment per Airline', fontsize=18)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
chakin.search(lang='English')

chakin.download(number=21, save_dir='./')

def divide_data(X,Y):

    """

    This code divide data into training, validation, and testing datasets

    """

     # Divide all tweets to 80%-20% training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        Y,

        test_size=0.2,

        shuffle=True,

        stratify = Y,

        random_state=42)



    # Divide the training data to 80%-20% training and validation sets

    X_train, X_valid, y_train, y_valid = train_test_split(

        X_train,

        y_train,

        test_size=0.2,

        shuffle=True,

        stratify = y_train,

        random_state=42)

    return  X_train, X_valid, X_test, y_train, y_valid, y_test
def create_LSTM_model(max_fatures, out_size, input_len, embedding_matrix):

    model = Sequential()

    model.add(Embedding(input_dim=max_fatures, output_dim=out_size, 

                        input_length=input_len, weights = [embedding_matrix], trainable=False))

    model.add(Dense(10))

    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))

    model.add(Dense(3, activation='softmax'))

    model.summary()



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



    return model
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


def evaluate(y_test, y_predict):

    roc = roc_auc_score(y_test, y_predict, average = 'weighted')

    accu = accuracy_score(y_test, y_predict)

    print ("Area Under Curve: ", roc)

    print ("Accurcy on test set: ", accu)

  

    

def deep_learning_tarining():

    """

    Data preparation: Prepare the tweets and embedding matrix that we will send to the neural network.

    """

    max_fatures = 2000                                                     

    tokenizer = Tokenizer(num_words=max_fatures, split=' ')          # use the most 2000 frequent words

    tokenizer.fit_on_texts(normalized_tweets)                        # create a vocabulary index

    all_tweets = tokenizer.texts_to_sequences(normalized_tweets)     # transform each review to a sequence of integers. Use only the most 200 frequent words

    all_tweets = pad_sequences(all_tweets, padding='post')           # pad short tweets with 0s

    

    # Preparing word embedding matrix for words in tweets

    vocab_size = len(tokenizer.word_index)+1

    embedding_matrix = np.zeros((vocab_size, 300))

    for  word, i in tokenizer.word_index.items():

        if word in word2vec.wv:

            embedding_vector = word2vec.wv[word]

            if embedding_vector is not None:

                embedding_matrix[i] = embedding_vector

            

    """

    - Divide the data into training, testing and validation sets

    - Convert catagorical output to one-hot vectors

    - Cast the classification problem as a multi-class classification task

    - Train the model for 20 epochs, and batch size of 200 instances

    

    """

    X = all_tweets

    Y = pd.get_dummies(airline_tweets['airline_sentiment']).values    # converts the catagorical outputs to one-hot vectors for each instance

    

    X_train, X_valid, X_test, y_train, y_valid, y_test = divide_data(X, Y)  # Split the data to three datasets

    

    model = create_LSTM_model(vocab_size, 300, X.shape[1], embedding_matrix)

    

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                                patience=3, 

                                                verbose=1, 

                                                factor=0.5, 

                                                min_lr=0.00001)

    

    history = model.fit(X_train, y_train, epochs = 20, batch_size = 200, validation_data=(X_valid, y_valid),callbacks = [learning_rate_reduction])



 

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.ylabel('accurcy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



    y_test_pred = model.predict(X_test)

    predictions = (y_test_pred == y_test_pred.max(axis=1, keepdims=1)).astype(int)

    evaluate(y_test, predictions)

    
deep_learning_tarining()
def decision_tree_training():

    splitted_tweets = []

    for tweet in normalized_tweets:

        bigramFeatureVector = []

        tweet = tweet.split()

        if len(tweet) == 1:

            splitted_tweets.append(str(tweet))

        else:

            for item in nltk.bigrams(tweet):

                bigramFeatureVector.append(' '.join(item))    

            splitted_tweets.append(str(bigramFeatureVector))

    

  

    X = splitted_tweets

    Y = pd.get_dummies(airline_tweets['airline_sentiment']).values              # converts the catagorical outputs to one-hot vectors for each instance



    X_train, X_valid, X_test, y_train, y_valid, y_test = divide_data(X, Y)      # Split the data to three datasets

    

    vectorizer = CountVectorizer(max_features=2000, min_df=5, max_df=0.8)       # Create CountVectorizer object that would tokenize a collection of text documents and build a vocabulary of known words

                                                                                # It would Keep only the most 2000 frequent terms that have occured at least 5 times, and at most in 80% of documents. 

            

                                                                                

                                                                                

    X_train = vectorizer.fit_transform(X_train).toarray()                       # Convert the tweets to a matrix of terms(features) counts, each row corrosponds to a tweet

                                                                                # and each column corrosponds to a word in the vocabulary. The fit part learns the vocabulary based on the CountVectorizer parameters

                                                                                # and the transform part encodes each tweet into a vector.

    X_test = vectorizer.transform(X_test).toarray() 

        

    text_classifier = RandomForestClassifier(n_estimators=200, max_features= 200, random_state=0)  

    text_classifier.fit(X_train, y_train) 

    

       

    predictions = text_classifier.predict(X_test)  

    evaluate(y_test, predictions)
decision_tree_training()