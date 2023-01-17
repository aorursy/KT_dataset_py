import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import *

from tensorflow.keras.models import *

import os

data=pd.read_csv("../input/entity-annotated-corpus/ner.csv",encoding="latin-1",error_bad_lines=False)

data.head()

data=data.iloc[:,-4:]

data=data.drop(["shape"],axis=1)

D=data.groupby("sentence_idx").apply(lambda x:((list(x["word"]),list(x["tag"]))))

data.dropna(inplace=True)

X=[]

Y=[]

dic={i:u for u,i in enumerate(set(data["word"]))}

tag_dic={j:u for u,j in enumerate(set(data["tag"]))}

dic

D[1.0]
X=[]

Y=[]

for i,j in D:

    for a in range(len(i)):

        i[a]=dic.get(i[a])

    for b in range(len(j)):

        j[b]=tag_dic.get(j[b])

    X.append(i)

    Y.append(j)
sns.distplot([len(i) for i in X])


X_train=pad_sequences(X,maxlen=100,padding="post",truncating="post")

Y_train=pad_sequences(Y,maxlen=100,padding="post",truncating="post",value=tag_dic.get("O"))

Y_train[1]
print(X_train.shape)

print(Y_train.shape)
Y_train=tf.keras.utils.to_categorical(Y_train,num_classes=len(tag_dic)+1)

Y_train.shape
I=Input(shape=(100,))

E=Embedding(len(dic)+1,100)(I)

B1=Bidirectional(LSTM(64,return_sequences=True))(E)

B2=Bidirectional(LSTM(64,return_sequences=True))(B1)

B3=LSTM(128,return_sequences=True)(B2)

D1=Dense(128,activation="relu")(B3)

D2=Dense(64,activation="relu")(D1)

O=Dense(18,activation="softmax")(D2)

model=Model(I,O)

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.summary()

model.fit(X_train,Y_train,epochs=2,validation_split=0.2)