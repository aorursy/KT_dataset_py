!pip install pymorphy2
# !pip install --force-reinstall tensorflow
# !pip install --ignore-installed --upgrade tensorflow==1.14.0
# !pip install --upgrade pip setuptools wheel
!pip install -I tensorflow
# !pip install -I keras

import nltk as nltk
import pandas as pd
import pymorphy2
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


analyzer = pymorphy2.MorphAnalyzer()

def get_normal_form_of_single_text(text):
    normalized_text = ""
    tokens = nltk.word_tokenize(text)
    
    for token in tokens:            
        normalized_text = normalized_text + " " + analyzer.parse(token)[0].normal_form
    return normalized_text
df = pd.read_csv("../input/allreviews/all-reviews.csv", names=["name", "review", "label"], skiprows= 1)
df['label'] = df['label'].map(lambda x: "1" if x not in ["-1", "0", "1"] else x)

df['review'] = df['review'].map(lambda x: get_normal_form_of_single_text(x))
my_films = ["Побег из Шоушенка", "Поймай меня, если сможешь", "Престиж"]
my_reviews = df[df.name.isin(my_films)]
train_reviews = df[~df.name.isin(my_films)]
my_reviews.head()
tfidf = TfidfVectorizer(max_features=500)

train_X_Tfidf = tfidf.fit_transform(train_reviews["review"])
test_X_Tfidf = tfidf.transform(my_reviews["review"])
print(test_X_Tfidf)
from keras.layers import Activation

train_Y_keras = to_categorical(train_reviews["label"], num_classes = 3)
test_Y_keras = to_categorical(my_reviews["label"],  num_classes = 3)

model = Sequential()
model.add(Dense(512, input_shape=(500,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(metrics=["accuracy"], optimizer='adam', loss='categorical_crossentropy')

model.fit(train_X_Tfidf, train_Y_keras, epochs=1000, batch_size=32)
result = model.predict(test_X_Tfidf)
print(classification_report(test_Y_keras.argmax(axis=1), result.argmax(axis=1)))