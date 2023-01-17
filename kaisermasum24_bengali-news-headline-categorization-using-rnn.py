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
import pandas as pd

import itertools

import os

import numpy as np # linear algebra

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf



from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.metrics import confusion_matrix



from tensorflow import keras

import keras.utils as ku

from keras.callbacks import EarlyStopping

from keras.utils.vis_utils import plot_model

from keras.layers import Dropout

from keras import optimizers



from numpy.random import seed





layers = keras.layers

models = keras.models
df = pd.read_excel("/kaggle/input/bengali-news-headline-categories/Bengali_News_Headline.xlsx",encoding='utf-8')
df.head()
for i in range(5):

    print("News:",i+1)

    print("Text:",df.Headline[i])

    print("NewsType:",df.NewsType[i])
contractions = { 

"বি.দ্র ": "বিশেষ দ্রষ্টব্য",

"ড.": "ডক্টর",

"ডা.": "ডাক্তার",

"ইঞ্জি:": "ইঞ্জিনিয়ার",

"রেজি:": "রেজিস্ট্রেশন",

"মি.": "মিস্টার",

"মু.": "মুহাম্মদ",

"মো.": "মোহাম্মদ",

}
import string

import re

def clean_text(text,remove_stopwords = False):

    if True:

        text = text.split()

        new_text = []

        for word in text:

            if word in contractions:

                new_text.append(contractions[word])

            else:

                new_text.append(word)

        text = " ".join(new_text)

    # Format words and remove unwanted characters

    whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)

    bangla_digits = u"[\u09E6\u09E7\u09E8\u09E9\u09EA\u09EB\u09EC\u09ED\u09EE\u09EF]+"

    english_chars = u"[a-zA-Z0-9]"

    punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"

    bangla_fullstop = u"\u0964"     #bangla fullstop(dari)

    

    punctSeq   = u"['\"“”‘’]+|[.?!,…]+|[:;]+"

    

    text = re.sub(bangla_digits, " ", text)

    text = re.sub(punc, " ", text)

    text = re.sub(english_chars, " ", text)

    text = re.sub(bangla_fullstop, " ", text)

    text = re.sub(punctSeq, " ", text)

    text = whitespace.sub(" ", text).strip()

    

   

            

    return text
clean_type = []

for newstype in df.NewsType:

    clean_type.append(clean_text(newstype,remove_stopwords=True))



clean_texts = []

for text in df.Headline:

    clean_texts.append(clean_text(text))
df['NewsType'].value_counts()
train_size = int(len(df) * .80)

print ("Train size: %d" % train_size)

print ("Test size: %d" % (len(df) - train_size))
def train_test_split(df, train_size):

    train = df[:train_size]

    test = df[train_size:]

    return train, test
train_cat, test_cat = train_test_split(df['NewsType'], train_size,)

train_text, test_text = train_test_split(df['Headline'], train_size,)
max_words = 5000

tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, 

                                              char_level=False)
tokenize.fit_on_texts(train_text)

x_train = tokenize.texts_to_matrix(train_text)

x_test = tokenize.texts_to_matrix(test_text)
encoder = LabelEncoder()

encoder.fit(train_cat)

y_train = encoder.transform(train_cat)

y_test = encoder.transform(test_cat)
num_classes = np.max(y_train) + 1

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)

print('y_train shape:', y_train.shape)

print('y_test shape:', y_test.shape)
batch_size = 16

epochs = 50

adam=keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
model = models.Sequential()

model.add(layers.Dense(16,kernel_initializer='uniform',input_shape=(max_words,)))

model.add(layers.Dropout(0.2))

model.add(layers.Activation('relu'))

model.add(layers.Dense(num_classes))

model.add(layers.Dropout(0.2))

model.add(layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

model.summary()
history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=2,

                    validation_split=0.2)
score = model.evaluate(x_train, y_train,

                       batch_size=batch_size, verbose=2)

print('Train loss:', score[0])

print('Train accuracy:', score[1])
score = model.evaluate(x_test, y_test,

                       batch_size=batch_size, verbose=2)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
def plot_history(history):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)

    

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training accuracy')

    plt.plot(x, val_acc, 'r', label='Validation accuracy')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

plot_history(history)
text_labels = encoder.classes_ 

for i in range(20):

    prediction = model.predict(np.array([x_test[i]]))

    predicted_label = text_labels[np.argmax(prediction)]

    print(test_text.iloc[i][:50], "...")

    print('Actual NewsType:' + test_cat.iloc[i])

    print("Predicted NewsType: " + predicted_label + "\n") 