# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.




# import needed libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re



import nltk 

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer





from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,accuracy_score  # Perform classification with SVM, kernel=linear

#reading data

data = pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")
# the shape of the data

data.shape
# columns and corresponding data types and null values ...

data.info()
# take a  samples of the data

data.head()
sentiment_counts = data.airline_sentiment.value_counts()

number_of_tweets = data.tweet_id.count()

print(sentiment_counts)
# visualize the values of airline_sentiment and count for each value ...

print(data['airline_sentiment'].value_counts())

data['airline_sentiment'].value_counts().plot(kind='bar')
# visualize the airlines and count for eache one

print(data['airline'].value_counts())

data['airline'].value_counts().plot(kind='bar')
# visualize the airlines with airline_sentiment

data.groupby(['airline', 'airline_sentiment']).size().unstack().plot(kind='bar')
# visualize the 'negativereason' values counts

data['negativereason'].value_counts().plot(kind='bar')
# The 'negative_reason' feature  have value when the sentiment is negative, and NAN for other sentiments.

data.groupby(['negativereason', 'airline_sentiment']).size().unstack().plot(kind='bar')
# preprocessing ...

""" 

-Remove punctuations, special characters. (only letters still in the text)

-Tokenizing text

-Convert words to lower case

-Remove stopwords

-Lemmatization

"""



# from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

wordnet_lemmatizer = WordNetLemmatizer()

def clean_data(tweet):

    only_letters = re.sub("[^a-zA-Z]", " ", tweet)

    tokenized_words = only_letters.split()

    # to lower case

    words_lc = [l.lower() for l in tokenized_words]

    # remove stopwords

    clean_words = [w for w in words_lc if w not in stop_words]



    lemmatize_words = [wordnet_lemmatizer.lemmatize(t) for t in clean_words]

    final_text=''.join(w+" " for w in lemmatize_words)

    return final_text



# add new columns ...

data['cleaned_tweet'] = data.text.apply(clean_data)

data[['text', 'cleaned_tweet']].head()





X=data.cleaned_tweet.values

# mapping each sentiment to integer label

def sentiment_to_label(sentiment):

    return {

        'negative': 0,

        'neutral': 1,

        'positive': 2

    }[sentiment]





Y = data.airline_sentiment.apply(sentiment_to_label).values
#split data to train, test

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=0)



print("X_train:" ,X_train.shape)

print("X_test: ",X_test.shape)

#sample of X_traing data 

X_train
# define and apply CountVectorizer on training data

cv = CountVectorizer(analyzer = "word")

train_features= cv.fit_transform(X_train)

test_features=cv.transform(X_test)
test_features.shape
#convert from sparse (contain a lot of zeros) to dense

train_features_2_array=train_features.toarray()

test_features_2_array= test_features.toarray()

print(train_features_2_array.shape)

print(test_features_2_array.shape)
#SVM model

print("Training SVM model  based on CountVectorizer ..")

svm_classifier = svm.SVC(kernel="rbf", C=0.025, probability=True)

svm_classifier.fit(train_features_2_array, Y_train)



svm_tr_prediction = svm_classifier.predict(train_features_2_array)

svm_ts_prediction = svm_classifier.predict(test_features_2_array)
print(" SVM OUTPUT USING CountVectorizer  ... \n\n")



svm_tr_accuracy = accuracy_score(svm_tr_prediction, Y_train)

print(" SVM training accuracy : ", svm_tr_accuracy)



svm_ts_accuracy = accuracy_score(svm_ts_prediction ,Y_test)

print(" SVM testing accuracy : ", svm_ts_accuracy)
svm_report = classification_report(Y_test, svm_ts_prediction)

print(svm_report)
print('training Logistic Regression model based on CountVectorizer')

LR_classifier = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=250)

LR_classifier.fit(train_features_2_array, Y_train)



LR_tr_prediction = LR_classifier.predict(train_features_2_array)

LR_ts_prediction = LR_classifier.predict(test_features_2_array)

LR_tr_accuracy = accuracy_score(LR_tr_prediction,Y_train)

print(" Logistic Regression training accuracy is: " ,LR_tr_accuracy)



LR_ts_accuracy = accuracy_score(LR_ts_prediction,Y_test)

print(" Logistic Regression testing accuracy is: ",LR_ts_accuracy)
svm_report = classification_report(Y_test, LR_ts_prediction)

print(svm_report)
# define and apply TfidfVectorizer on training data



tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(X_train)

train_tfidf_features =  tfidf_vect.transform(X_train)

test_tfidf_features =  tfidf_vect.transform(X_test)

train_tfidf_features_2_array=train_features.toarray()

test_tfidf_features_2_array= test_features.toarray()
print("Training SVM model  based on TfidfVectorizer ..")



svm_classifier = svm.SVC(kernel="rbf", C=0.025, probability=True)

svm_classifier.fit(train_tfidf_features_2_array, Y_train)



svm_tr_prediction = svm_classifier.predict(train_tfidf_features_2_array)

svm_ts_prediction = svm_classifier.predict(test_tfidf_features_2_array)
print(" SVM OUTPUT USING TfidfVectorizer ... \n\n")



svm_tr_accuracy = accuracy_score(svm_tr_prediction, Y_train)

print(" SVM training accuracy : ", svm_tr_accuracy)



svm_ts_accuracy = accuracy_score(svm_ts_prediction,Y_test)

print(" SVM testing accuracy : ", svm_ts_accuracy)

report = classification_report(Y_test, svm_ts_prediction)

print(report)
print('training Logistic Regression model based on TfidfVectorizer ..')

LR_classifier = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200)

LR_classifier.fit(train_tfidf_features_2_array, Y_train)



LR_tr_predictions = LR_classifier.predict(train_tfidf_features_2_array)

LR_ts_predictions = LR_classifier.predict(test_tfidf_features_2_array)

print(" Logistic Regression OUTPUT USING TfidfVectorizer ... \n\n")

LR_tr_accuracy = accuracy_score(LR_tr_predictions,Y_train)

print(" Logistic Regression Train accuracy is: " ,LR_tr_accuracy)



LR_ts_accuracy = accuracy_score(LR_ts_predictions,Y_test)

print(" Logistic Regression Test accuracy is: ",LR_ts_accuracy)
report = classification_report(Y_test, LR_ts_predictions)

print(report)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



from tensorflow.keras.layers import Embedding

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import LSTM
embeddings_index = {}

f = open('/kaggle/input/glove-embedding-vectors/glove.840B.300d.txt')



for line in f:

    values = line.split(' ')

    word = values[0] ## The first entry is the word

    coefs = np.asarray(values[1:], dtype='float32') 

    embeddings_index[word] = coefs

f.close()



print('GloVe data loaded')

print('Loaded %s word vectors.' % len(embeddings_index))

#encode train texts and test texts using the a tokenizer

MAX_NUM_WORDS = 1000

MAX_SEQUENCE_LENGTH = 135 #from the stats we found previously

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(X)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



sequences_train = tokenizer.texts_to_sequences(X_train)

X_train_seq = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)



sequences_test = tokenizer.texts_to_sequences(X_test)

X_test_seq = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)



#convert labels to one hot vectors

labels_train = to_categorical(np.asarray(Y_train))

labels_test = to_categorical(np.asarray(Y_test))



print("train data :")

print(X_train_seq.shape)

print(labels_train.shape)



print("test data :")

print(X_test_seq.shape)

print(labels_test.shape)
# Find number of unique words in our tweets

vocab_size = len(word_index) + 1 

# Define size of embedding matrix: number of unique words x embedding dim (300)

embedding_matrix = np.zeros((vocab_size, 300))



# fill in matrix

for word, i in word_index.items():  # dictionary

    embedding_vector = embeddings_index.get(word) # gets embedded vector of word from GloVe

    if embedding_vector is not None:

        # add to matrix

        embedding_matrix[i] = embedding_vector # each row of matrix

#DL model: pass the encoded data to an embedding layer and use the Glove pre_trained weights, then pass the 

# output to an LSTM layer follwed by 2 dense layers.

# the optimizer used is Adam, since it achivied higher accurcies usually.



cell_size= 256

deepLModel1 = Sequential()

embedding_layer = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],

                           input_length = MAX_SEQUENCE_LENGTH, trainable=False)

deepLModel1.add(embedding_layer)

deepLModel1.add(LSTM(cell_size, dropout = 0.2))

deepLModel1.add(Dense(64,activation='relu'))

deepLModel1.add(Flatten())

deepLModel1.add(Dense(3, activation='softmax'))

deepLModel1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

deepLModel1.summary()
#train the model

deepLModel1_history = deepLModel1.fit(X_train_seq, labels_train, validation_split = 0.25,

                    epochs=50, batch_size=256)
# Find train and test accuracy

loss, accuracy = deepLModel1.evaluate(X_train_seq, labels_train, verbose=False)

print("Training Accuracy: ",accuracy)



loss, accuracy = deepLModel1.evaluate(X_test_seq, labels_test, verbose=False)

print("Testing Accuracy: ",accuracy)





predictions_test = deepLModel1.predict_classes(X_test_seq)

#print other performance measures, espically the data is unbalanced

print(classification_report(predictions_test , Y_test))


