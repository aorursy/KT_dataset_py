import numpy as np

import pandas as pd

import os

import re

import seaborn as sns

import keras



from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score



from wordcloud import WordCloud, STOPWORDS 
data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

reviews = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv')
reviews
reviews['Sentiment'].notna().count()
reviews['Sentiment'].value_counts()
reviews['Sentiment'].dropna(axis='rows');
reviews['Sentiment'].isna().count()
reviews
reviews = reviews[reviews['Translated_Review'].notna()]
lemmatizer = WordNetLemmatizer()

max_words = 0

uniq_words = set()



def clean_text(value):

    global max_words, uniq_words

    val = re.sub('[^a-zA-Z0-9]', ' ', value)

    

    vals = [lemmatizer.lemmatize(w) for w in word_tokenize(val.lower())]

    

    if len(vals) > max_words:

        max_words = len(vals)



    uniq_words = uniq_words.union(vals)

    return ' '.join(vals)
reviews['Translated_Review_tokens'] = reviews['Translated_Review'].map(clean_text)

num_uniq_words = len(uniq_words)
def show_word_cloud(text):

    stopwords = set(STOPWORDS) 



    wordcloud = WordCloud(

        width = 800,

        height = 800,

        background_color ='white',

        stopwords = stopwords,

        min_font_size = 10

    ).generate(text)

       

    plt.figure(figsize = (8, 8), facecolor = None)

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.tight_layout(pad = 0)

    plt.show()
show_word_cloud(reviews['Translated_Review_tokens'].str.cat(sep=' '))
tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_uniq_words)

tokenizer.fit_on_texts(reviews['Translated_Review_tokens'])
reviews['Sentiment'] = reviews['Sentiment'].astype('category')
x_train_reviews, x_validate_reviews, y_train_reviews, y_validate_reviews = train_test_split(

    reviews['Translated_Review_tokens'],

    reviews['Sentiment'],

    test_size=0.2,

    random_state=42,

)
num_classes = 3

batch_size=100
x_train_reviews_bin = tokenizer.texts_to_matrix(x_train_reviews, mode='binary')

x_validate_reviews_bin = tokenizer.texts_to_matrix(x_validate_reviews, mode='binary')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



y_train_reviews = le.fit_transform(y_train_reviews)

y_validate_reviews = le.fit_transform(y_validate_reviews)
y_train_reviews_ohe = keras.utils.to_categorical(y_train_reviews, num_classes)

y_validate_reviews_ohe = keras.utils.to_categorical(y_validate_reviews, num_classes)
model = keras.Sequential()

model.add(keras.layers.Dense(186, input_shape=(num_uniq_words,)))

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_classes))

model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(monitor='value_loss')



history = model.fit(

    x_train_reviews_bin,

    y_train_reviews_ohe,

    epochs=10,

    batch_size=batch_size,

    verbose=1,

    validation_split=0.1,

    callbacks=[early_stopping],

)
score = model.evaluate(x_validate_reviews_bin, y_validate_reviews_ohe, batch_size=batch_size)

print(score)
print(f'loss: {score[0]}, accuracy: {score[1]}')
y_validate_predicted = model.predict(x_validate_reviews_bin)

y_validate_predicted = y_validate_predicted.astype('int')
prec = precision_score(y_validate_predicted, y_validate_reviews_ohe, average="macro")

print(f'precision: {prec}')
x_train_reviews_seq = tokenizer.texts_to_sequences(x_train_reviews)

x_validate_reviews_seq = tokenizer.texts_to_sequences(x_validate_reviews)
x_train_reviews_seq = keras.preprocessing.sequence.pad_sequences(x_train_reviews_seq, maxlen=max_words)

x_validate_reviews_seq = keras.preprocessing.sequence.pad_sequences(x_validate_reviews_seq, maxlen=max_words)
model = keras.Sequential()

model.add(keras.layers.Embedding(num_uniq_words, 250, mask_zero=True))

model.add(keras.layers.LSTM(128,dropout=0.4, recurrent_dropout=0.4, return_sequences=True))

model.add(keras.layers.LSTM(64,dropout=0.5, recurrent_dropout=0.5, return_sequences=False))

model.add(keras.layers.Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])

model.summary()
history = model.fit(

    x_train_reviews_seq,

    y_train_reviews_ohe,

    validation_data=(x_validate_reviews_seq, y_validate_reviews_ohe),

    epochs=5,

    batch_size=batch_size,

    verbose=1,

    callbacks=[early_stopping],

)
model.save('lstm_model.h5')
y_validate_predicted = model.predict_classes(x_validate_reviews_seq)

y_validate_predicted = y_validate_predicted.astype('int')
acc = accuracy_score(y_validate_predicted, y_validate_reviews.astype('int'))

prec = precision_score(y_validate_predicted, y_validate_reviews.astype('int'), average="macro")

print(f'accuracy: {acc}, precision: {prec}')