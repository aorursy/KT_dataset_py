import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D



import os

from zipfile import ZipFile



tf.__version__
train_pos_dir = '/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/pos'

train_neg_dir = '/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/neg'

test_pos_dir = '/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/pos'

test_neg_dir = '/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/neg'
list_of_training_files = os.listdir(train_pos_dir) + os.listdir(train_neg_dir)

list_of_training_files[:4]
# creating a dataframe from the files

rating_list = []

review_list = []

sentiment_list = []



for filename in list_of_training_files:

    rate = int(filename.split('_')[-1].split('.')[-2])

    rating_list.append(rate)

    if(rate >= 5):

        sentiment_list.append(1) # positive sentiment

        review_list.append(open(train_pos_dir + '/' + filename).read())

    else:

        sentiment_list.append(-0) # negative sentiment

        review_list.append(open(train_neg_dir + '/' + filename).read())

data_train = pd.DataFrame({'Reviews':review_list, 'Rates':rating_list, 'Sentiments':sentiment_list})

data_train.head()
max_features = 220
# Creating Tokens from the text

tokenizer = Tokenizer(num_words = max_features, split = (' '))
tokenizer.fit_on_texts(data_train['Reviews'].values)
X = tokenizer.texts_to_sequences(data_train['Reviews'].values)

# making all the tokens into same sizes using padding.

X = pad_sequences(X, maxlen = max_features)
X.shape
Y = data_train['Sentiments'].values
# creating a Deep Learning model

model = Sequential()

model.add(Embedding(max_features, 64, input_length = X.shape[1]))

model.add(SpatialDropout1D(rate = 0.2))

model.add(LSTM(64, dropout = 0.2, recurrent_dropout = 0.2))

model.add(Dense(2, activation = 'softmax'))



model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X, Y, epochs = 5)
test_pos_dir
# creating the testing dataframe



list_of_testing_filenames = os.listdir(test_pos_dir) + os.listdir(test_neg_dir)



# creating a dataframe from the files

rating_list = []

review_list = []

sentiment_list = []



for filename in list_of_testing_filenames:

    rate = int(filename.split('_')[-1].split('.')[-2])

    rating_list.append(rate)

    if(rate >= 5):

        sentiment_list.append(1) # positive sentiment

        review_list.append(open(test_pos_dir + '/' + filename).read())

    else:

        sentiment_list.append(-0) # negative sentiment

        review_list.append(open(test_neg_dir + '/' + filename).read())

data_test = pd.DataFrame({'Reviews':review_list, 'Rates':rating_list, 'Sentiments':sentiment_list})

data_test.head()
data_test['Sentiments'].nunique()
# using the  tokenizer

tokenizer = Tokenizer(num_words = max_features, split = (' '))

X_test = tokenizer.texts_to_sequences(data_test['Reviews'].values)

# making all the tokens into same sizes using padding.

X_test = pad_sequences(X, maxlen = max_features)
Y_test = data_test['Sentiments'].values
# predicting the testing dataframe

prediction = model.predict_classes(X_test)
# metrics for calculating the performance of the model

from sklearn.metrics import classification_report
print(classification_report(prediction, Y_test))
def prediction_unseen_review(review):

    token = Tokenizer(num_words = max_features, split = (' '))

    token.fit_on_texts(review)

    X_unseen = token.texts_to_sequences(review)

    X_unseen = pad_sequences(X_unseen, maxlen = max_features)

    prediction_unseen = model.predict_classes(X_unseen)

    if (prediction_unseen.mean() >= 0.5):

        print(prediction_unseen.mean())

        print('Positive')

    else:

        print(prediction_unseen.mean())

        print('Negative')
prediction_unseen_review('This movie is fantastic')