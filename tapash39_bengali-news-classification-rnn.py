import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/bengali-news-dataset/train.csv')

train.head()
train.describe()
test = pd.read_csv('/kaggle/input/bengali-news-dataset/valid.csv')

test.head()
# fix random seed for reproducibility

np.random.seed(7)

train = train.drop_duplicates().reset_index(drop=True)

test = test.drop_duplicates().reset_index(drop=True)
train.label.unique()
train.label = train.label.replace('entertainment', 1)

train.label = train.label.replace('national', 2)

train.label = train.label.replace('sports', 3)

train.label = train.label.replace('kolkata', 4)

train.label = train.label.replace('state', 5)

train.label = train.label.replace('international', 6)

train.label = train.label.replace('sport', 7)

train.label = train.label.replace('nation', 8)

train.label = train.label.replace('world', 9)

train.label = train.label.replace('travel', 10)

train.label.head()
test.label.unique()
test.label = test.label.replace('entertainment', 1)

test.label = test.label.replace('national', 2)

test.label = test.label.replace('sports', 3)

test.label = test.label.replace('kolkata', 4)

test.label = test.label.replace('state', 5)

test.label = test.label.replace('international', 6)

test.label = test.label.replace('sport', 7)

test.label = test.label.replace('nation', 8)

test.label = test.label.replace('world', 9)

test.label = test.label.replace('travel', 10)

train.label.head()
train = train.append(test)

df = train

df.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer

from keras.layers import SpatialDropout1D

from keras.utils import to_categorical

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping

from sklearn.feature_selection import RFE

import re
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 50000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 250

# This is fixed.

EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~।', lower=False)

tokenizer.fit_on_texts(df.article.values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(df.article.values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)
Y = pd.get_dummies(df.label).values

print('Shape of label tensor:', Y.shape)
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=.10)


model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
epochs = 5

batch_size = 32



history = model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
accr = model.evaluate(test_features,test_labels)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
from matplotlib import pyplot as plt

plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()

plt.title('Accuracy')

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.show()
model.save_weights('bengali_news_model.h5')
news = ["""‘বন্ধুত্ব’ করিয়ে দেওয়ার টোপ দিয়ে টাকা হাতানোর অভিযোগে ১৬ জনকে গ্রেফতার করল কলকাতা পুলিশের সাইবার ক্রাইম থানা। ধৃতদের মধ্যে ন’জন পুরুষ এবং সাতজন মহিলা। তাদের মঙ্গলবার ব্যাঙ্কশাল আদালতে তোলা হলে বিচারক ১৫ মে পর্যন্ত পুলিশি হেফাজতের নির্দেশ দিয়েছেন।\

    পুলিশ জানিয়েছে, অভিযুক্তেরা বিভিন্ন সংবাদপত্রে ‘এসকর্ট সার্ভিসে’র বিজ্ঞাপন দিত। মহিলাদের সঙ্গে ‘বন্ধুত্ব’ করে লোভনীয় উপার্জনের হাতছানি থাকত সেইসব বিজ্ঞাপনে। এই কাজের জন্য গাড়ি করে নিয়ে যাওয়া এবং বাড়িতে পৌঁছে দেওয়ার ব্যবস্থাও আছে বলে লেখা থাকত সেখানে। যোগাযোগের জন্য দু’টি মোবাইল নম্বরও দেওয়া থাকত। ওই নম্বরে যোগাযোগ করলে ব্যাঙ্ক অ্যাকাউন্টে একাধিকবার বিভিন্ন খাতে টাকা জমা দিতে বলা হতো।

    অ্যাকাউন্টে টাকা পৌঁছে গেলেই সাইবার ক্রাইম থানার পুলিশ আধিকারিক পরিচয় দিয়ে প্রতারণাচক্রের এক ব্যক্তি ফোন করত সংশ্লিষ্ট আবেদনকারীকে। ফোনে গ্রেফতারির হুমকির পাশাপাশি, ২০ হাজার টাকা দাবি করা হতো বলে অভিযোগ। \

    এই প্রতারণাচক্রের খপ্পরে পড়া মুচিপাড়া এলাকার এক বাসিন্দা এপ্রিল মাসে পুলিশের কাছে অভিযোগ দায়ের করেছিলেন। অভিযোগের ভিত্তিতে তদন্ত করতে গিয়ে কসবার রাজডাঙায় একটি ফ্ল্যাটের সন্ধান মেলে। সোমবার রাতে সেখানে হানা দিয়ে অভিযুক্তদের গ্রেফতার করা হয়। ধৃতদের কাছ থেকে ৫৩টি মোবাইল ফোন, ৭৫টি সিম কার্ড, দু’টি কর্ডলেস ফোন, একাধিক রাবার স্ট্যাম্প এবং একটি গাড়ি বাজেয়াপ্ত করেছে পুলিশ।  

       """]

seq = tokenizer.texts_to_sequences(news)

padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

pred = model.predict(padded)

labels = ['entertainment', 'national', 'sports', 'kolkata', 'state','international', 'sport', 'nation', 'world', 'travel']

label = pred, labels[np.argmax(pred)]

print("News Label Is: ")

print(label[1])