# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

path_list = []

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path_list.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path_list
train = pd.read_csv(path_list[0])
xtrain,ytrain = train.iloc[:,3],train.iloc[:,4]
xtrain = list(xtrain)
x_train = []

for i in xtrain:

    x_train.append(i.lower())
del(xtrain)
x_train[0:20]
import re

def preprocess(string):

    string = str(string)

    string = string.lower()

    string = string.replace("_"," ")

    string = re.sub('^(http:\/\/www\.|https:\/\/www\.|(http|htt):\/\/|https:\/\/)?[a-z\@0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?(\w*\W*)$' , ' ', string)

    string = re.sub(r"\d*[^\w^\,^\']", ' ', string)

    string = string.replace(',',' , ')

    string = string.replace('  ',' ')

    string = string.replace('  ',' ')

    pattern = re.compile(r'(\w)(\1{2,})')

    string = pattern.sub(r"\1\1", string)

    string = string.lstrip(' ')

    string  = string.rstrip(' ')

    return string
xtrain = []

for i in x_train:

    xtrain.append(preprocess(i))

del(x_train)
xtrain[:20]
import nltk

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df = 2)



vectorized_corpus = cv.fit_transform(xtrain)

len(cv.vocabulary_)
idx_word = dict([(keys,values) for (values,keys) in cv.vocabulary_.items()])
idx_word[6450]
del(cv.vocabulary_['0l'])
len(cv.vocabulary_)
cv.vocabulary_['?'] = 6450
len(cv.vocabulary_)
del(idx_word)
new_train = []

for i in xtrain:

    new = []

    for j in i.split():

        try:

            new.append(cv.vocabulary_[j])

        except:

            new.append(cv.vocabulary_['?'])

    new_train.append(new)
new_train[1]
del(xtrain)
ytrain[0]
import random

import numpy

c = list(zip(new_train, ytrain))



random.shuffle(c)



train, Label = zip(*c)

del(c)

print(type(train))

train = numpy.array(train)

Label = numpy.array(Label)

print(type(train))

del(new_train)

del(ytrain)
from keras.models import Sequential

from keras.layers import *
from keras.preprocessing import sequence



xtrain = sequence.pad_sequences(train,maxlen = 25)

del(train)
xtrain.shape
Label.shape
model = Sequential()

model.add(Embedding(6451,64))

model.add(LSTM(256,return_sequences = True))

model.add(Dropout(0.5))

model.add(LSTM(64,return_sequences = False))

model.add(Dense(1))

model.add(Activation("sigmoid"))

model.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("/kaggle/working/Best_model.h5",monitor = "val_loss",verbose = True,save_best_only=True)

earlystop = EarlyStopping(monitor = "val_acc",patience = 2)
model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ["acc"])
model.fit(xtrain,Label,epochs = 10,batch_size = 1024,shuffle = True,validation_split = 0.2,callbacks = [checkpoint,earlystop])
model.fit(xtrain,Label,epochs = 10,batch_size = 1024,shuffle = True,validation_split = 0.2,callbacks = [checkpoint,earlystop])
test = pd.read_csv(path_list[1])
test.tail()
test = list(test.iloc[:,3])
xtest = []

for i in test:

    xtest.append(preprocess(i))

del(test)

xtest[0]
new_test = []

for i in xtest:

    new = []

    for j in i.split():

        try:

            new.append(cv.vocabulary_[j])

        except:

            new.append(cv.vocabulary_['?'])

    new_test.append(new)
new_test[0]
xtest = sequence.pad_sequences(new_test,maxlen = 25)

del(new_test)
xtest[0]
pred = model.predict_classes(xtest)
result = pd.read_csv(path_list[2])
result.head()
pred = list(pred)
pred[0][0]
p = []

for i in pred:

    p.append(i[0])
p[0]
result["target"] = p
result.to_csv("/kaggle/working/result.csv")