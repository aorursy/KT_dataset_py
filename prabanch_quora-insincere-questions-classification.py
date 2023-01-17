# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import string

import re

from os import listdir

from numpy import array

from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer

from keras.utils.vis_utils import plot_model

from keras.models import Sequential

from keras.layers import Dense

from collections import Counter

# Scikit Learn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd

import numpy as np



import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS



from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_colwidth', -1)

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,roc_auc_score,roc_curve

import numpy as np

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from keras.preprocessing.sequence import pad_sequences



import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers





print("Libraries loaded")
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# Train_1 = train[train['target']==1]

# Train_0 = train[train['target']==0]

# train =  pd.concat([Train_1,Train_0.head(len(Train_1))], axis=0)
train
# train = train.head(10)?

test_df = test
train["target"].value_counts()
stopwords = set(STOPWORDS)





wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(train[train.target==0]['question_text']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=900)
stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(train[train.target==1]['question_text']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=900)
## split to train and val

train_df, val_df = train_test_split(train, test_size=0.1, random_state=2018)



## some config values 

embed_size = 100 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use



## fill up the missing values

train_X = train_df["question_text"].fillna("_na_").values

val_X = val_df["question_text"].fillna("_na_").values

test_X = test_df["question_text"].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values



# inp = Input(shape=(maxlen,))

# x = Embedding(max_features, embed_size)(inp)

# x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

# x = GlobalMaxPool1D()(x)

# x = Dense(16, activation="relu")(x)

# x = Dropout(0.1)(x)

# x = Dense(1, activation="sigmoid")(x)

# model = Model(inputs=inp, outputs=x)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# print(model.summary())

# model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))





# define the model

model = Sequential()

model.add(Embedding(max_features, embed_size, input_length=maxlen))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model

print(model.summary())

model.fit(train_X, train_y, batch_size=512, epochs=5, validation_data=(val_X, val_y))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,roc_auc_score,roc_curve, auc,  f1_score

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import f1_score

# Making the Confusion Matrix

def get_metrics(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)



    class_names=[0,1] # name  of classes

    fig, ax = plt.subplots()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)

    plt.yticks(tick_marks, class_names)

    # create heatmap

    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')

    ax.xaxis.set_label_position("top")

    plt.tight_layout()

    plt.title('Confusion matrix', y=1.1)

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')





    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



    # Model Precision: what percentage of positive tuples are labeled as such?

    print("Precision:",metrics.precision_score(y_test, y_pred))



    # Model Recall: what percentage of positive tuples are labelled as such?

    print(" True positive rate or (Recall or Sensitivity) :",metrics.recall_score(y_test, y_pred))



    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn+fp)



    #Specitivity. or True negative rate

    print(" True Negative rate or Specitivity :",specificity)



    false_negative = fn / (fn+tp)



    #False negative rate

    print(" False Negative rate :",false_negative)



    #False positive rate

    print(" False positive rate (Type 1 error) :",1 - specificity)

    

    print('F Score', f1_score(y_test, y_pred))

    print(cm)

val_pred_y = model.predict_classes([val_X], batch_size=1024, verbose=1)

get_metrics(val_y,val_pred_y)
test_y = model.predict_classes([test_X], batch_size=1024, verbose=1)

output = pd.DataFrame({"qid":test_df["qid"].values})

output['prediction'] = test_y

output.to_csv("submission.csv", index=False)