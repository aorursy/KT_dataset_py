%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd

pd.set_option('display.max_colwidth', -1)

from matplotlib import pyplot as plt

import seaborn as sns

import keras

from keras.models import load_model

from keras.layers import Dense, Embedding, LSTM, Dropout

from keras.models import Sequential

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

#import sqlite3

from tqdm import tqdm

import re

from bs4 import BeautifulSoup



from sklearn.model_selection import train_test_split
filtered_data = pd.read_csv("../input/instruments/reviews.csv")



# Give reviews with Score>3 a positive rating(1), and reviews with a score<3 a negative rating(0).

def partition(x):

    if x < 3:

        return 0

    return 1



def findMinorClassPoints(df):

    posCount = int(df[df['overall']==1].shape[0]);

    negCount = int(df[df['overall']==0].shape[0]);

    if negCount < posCount:

        return negCount

    return posCount



#changing reviews with score less than 3 to be positive and vice-versa

actualScore = filtered_data['overall']

positiveNegative = actualScore.map(partition) 

filtered_data['overall'] = positiveNegative



#Performing Downsampling

# samplingCount = findMinorClassPoints(filtered_data)

# postive_df = filtered_data[filtered_data['overall'] == 1].sample(n=5000)

# negative_df = filtered_data[filtered_data['overall'] == 0].sample(n=5000)



# filtered_data = pd.concat([postive_df, negative_df])



print("Number of data points in our data", filtered_data.shape)

filtered_data.head(3)


#Deduplication of entries

final=filtered_data.drop_duplicates(subset={"reviewerID","asin","reviewerName","reviewText","unixReviewTime"}, keep='first', inplace=False)

final.shape



#Preprocessing

def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase





preprocessed_reviews = []

# tqdm is for printing the status bar

for sentence in tqdm(final['reviewText'].values):

    sentence = re.sub(r"http\S+", "", str(sentence))

    sentence = BeautifulSoup(sentence, 'lxml').get_text()

    sentence = decontracted(sentence)

    sentence = re.sub("\S*\d\S*", "", sentence).strip()

    sentence = re.sub('[^A-Za-z]+', ' ', sentence)

    # https://gist.github.com/sebleier/554280

    # sentance = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)

    preprocessed_reviews.append(sentence.strip())

    

## Similartly you can do preprocessing for review summary also.

def concatenateSummaryWithText(str1, str2):

    return str1 + ' ' + str2



preprocessed_summary = []

# tqdm is for printing the status bar

for sentence in tqdm(final['summary'].values):

    sentence = re.sub(r"http\S+", "", str(sentence))

    #sentence = BeautifulSoup(sentence, 'lxml').get_text()

    sentence = decontracted(sentence)

    sentence = re.sub("\S*\d\S*", "", sentence).strip()

    sentence = re.sub('[^A-Za-z]+', ' ', sentence)

    # https://gist.github.com/sebleier/554280

    # sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)

    preprocessed_summary.append(sentence.strip())

    

preprocessed_reviews = list(map(concatenateSummaryWithText, preprocessed_reviews, preprocessed_summary))

final['CleanedText'] = preprocessed_reviews

final['CleanedText'] = final['CleanedText'].astype('str')
X = final['CleanedText']

y = final['overall']
del final

del preprocessed_reviews

del preprocessed_summary

del filtered_data
X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.30, stratify=y, shuffle=True)

X_train, X_cv, y_train, y_cv = train_test_split(X_t, y_t, test_size=0.30, stratify=y_t, shuffle=True)

print("Shape of Input  - Train:", X_train.shape)

print("Shape of Output - Train:", y_train.shape)

print("Shape of Input  - CV   :", X_cv.shape)

print("Shape of Output - CV   :", y_cv.shape)

print("Shape of Input  - Test :", X_test.shape)

print("Shape of Output - Test :", y_test.shape)
tokenize = Tokenizer(num_words=5000)

tokenize.fit_on_texts(X_train)



X_train_new = tokenize.texts_to_sequences(X_train)

X_cv_new = tokenize.texts_to_sequences(X_cv)

X_test_new = tokenize.texts_to_sequences(X_test)



print(X_train_new[1])

print(len(X_train_new))
# truncate and/or pad input sequences

max_review_length = 1000

X_train_new = sequence.pad_sequences(X_train_new, maxlen=max_review_length)

X_cv_new = sequence.pad_sequences(X_cv_new, maxlen=max_review_length)

X_test_new = sequence.pad_sequences(X_test_new, maxlen=max_review_length)



print(X_train_new.shape)

print(X_train_new[1])
# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4

# https://stackoverflow.com/a/14434334

# this function is used to update the plots for each epoch and error

def plt_dynamic(x, vy, ty, ax, colors=['b']):

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()

    fig.canvas.draw()

    

n_epochs = 5

batchsize = 512



final_output = pd.DataFrame(columns=["Model", "Architecture",

                                     "TRAIN_LOSS", "TEST_LOSS", "TRAIN_ACC", "TEST_ACC"]);
# create the model

embed_vector_length = 32

model = Sequential()

model.add(Embedding(5000, embed_vector_length, input_length=max_review_length))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("***********************************************")

print("Printing the Model Summary")

print(model.summary())

print("***********************************************")







model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

del model  # deletes the existing model



# returns a compiled model

# identical to the previous one

model = load_model('my_model.h5')
m_hist = model.fit(X_train_new, y_train, epochs=n_epochs, 

                   batch_size=batchsize, verbose=1, validation_data=(X_cv_new, y_cv))



score = model.evaluate(X_test_new, y_test, batch_size=batchsize)

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



final_output = final_output.append({"Model": 1,

                                    "Architecture": 'Embedding-LSTM-Sigmoid', 

                                    "TRAIN_LOSS": '{:.5f}'.format(m_hist.history["loss"][n_epochs-1]),

                                    "TEST_LOSS": '{:.5f}'.format(score[0]),

                                    "TRAIN_ACC": '{:.5f}'.format(m_hist.history["acc"][n_epochs-1]),

                                    "TEST_ACC": '{:.5f}'.format(score[1])}, ignore_index=True)



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch')

ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,n_epochs+1))



vy = m_hist.history['val_loss']

ty = m_hist.history['loss']

plt_dynamic(x, vy, ty, ax)