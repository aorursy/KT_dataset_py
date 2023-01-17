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
import bz2
bzfile = bz2.BZ2File('../input/test.ft.txt.bz2','r')
lines = bzfile.readlines()

lines[1]
sentimentlist = []
def splitdocs(docs,splitstr = '__label__'):
    for i in range(len(docs)):
        linetext = str(lines[i])
        splittedtext = linetext.split(splitstr)
        RHStext = splittedtext[1]
        sentiment = RHStext[0]
        n = len(RHStext)
        reviewtext = RHStext[2:n-1]
        sentimentlist.append([reviewtext,sentiment])
    print("done successfully")
        
    return sentimentlist
print(len(lines))
computedsentimentlist = []
computedsentimentlist = splitdocs(lines)
train_dataset = pd.DataFrame(computedsentimentlist,columns=['ReviewText','sentiment'])
train_dataset.head()
train_dataset['sentiment'][train_dataset['sentiment']=='1'] = 0
train_dataset['sentiment'][train_dataset['sentiment']=='2'] = 1
train_dataset.head()
train_dataset['word_count'] = train_dataset['ReviewText'].str.lower().str.split(' ').apply(len)
train_dataset.head()
import string 
def remove_punctuations(str1):
    table = str.maketrans({key: None for key in string.punctuation})
    return str1.translate(table)

train_dataset['ReviewText'] = train_dataset['ReviewText'].apply(remove_punctuations)
train_dataset.head()
train_dataset = train_dataset[train_dataset.word_count < 25]  
train_dataset
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
st_wd = text.ENGLISH_STOP_WORDS
c_vector = CountVectorizer(stop_words = st_wd,min_df=.0001,lowercase=1)
transformed_dataset = c_vector.fit_transform(train_dataset['ReviewText'].values)


from sklearn.model_selection import train_test_split
y = train_dataset['sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(transformed_dataset, y, test_size=0.1, random_state=42)
#converting sparse matrix to dense
X_train = X_train.todense()
X_test = X_test.todense()
X_train.shape

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model1=  Sequential()
model1.add(Dense(1000,input_shape=(8369,),activation='relu'))
model1.add(Dense(1,activation='sigmoid'))

model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist1 = model1.fit(X_train,y_train,epochs=6,batch_size=128,verbose=1)
#test accuracy
model1.evaluate(X_test, y_test, batch_size=128)
#train accuracy
model1.evaluate(X_train, y_train, batch_size=128)
#2 hidden layers
#first hidden layer has 1000 units and second hidden layer has 500 units
model2=  Sequential()
model2.add(Dense(1000,input_shape=(8369,),activation='relu'))
model2.add(Dense(500,activation='relu'))
model2.add(Dense(1,activation='sigmoid'))

model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist2 = model2.fit(X_train,y_train,epochs=6,batch_size=128,verbose=1)
# 2 hidden layer train accuracy
model2.evaluate(X_train, y_train, batch_size=128)
# 2 hidden layer test accuracy
model2.evaluate(X_test, y_test, batch_size=128)
model3=  Sequential()
model3.add(Dense(2000,input_shape=(8369,),activation='relu'))
model3.add(Dense(1000,activation='relu'))
model3.add(Dense(500,activation='relu'))
model3.add(Dense(1,activation='sigmoid'))

model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist3 = model3.fit(X_train,y_train,epochs=6,batch_size=128,verbose=1)
# 3 hidden layer train accuracy
model3.evaluate(X_train, y_train, batch_size=128)
# 3 hidden layer test accuracy
model3.evaluate(X_test, y_test, batch_size=128)
#plot of loss curve and accuarcy plot
import matplotlib.pyplot as plt
loss_curve1 = hist1.history['loss']
epoch_curve1 = list(range(len(loss_curve1)))
loss_curve2 = hist2.history['loss']
epoch_curve2 = list(range(len(loss_curve2)))
loss_curve3 = hist3.history['loss']
epoch_curve3 = list(range(len(loss_curve3)))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epoch_curve1,loss_curve1,label='1 Hidden layer')
plt.plot(epoch_curve2,loss_curve2,label='2 Hidden layers')
plt.plot(epoch_curve3,loss_curve3,label='3 Hidden layers')
plt.legend()
plt.show()
#accuracy curve
accuracy1 = hist1.history['acc']
epoch_curve1 = list(range(len(accuracy1)))
accuracy2= hist2.history['acc']
epoch_curve2 = list(range(len(accuracy2)))
accuracy3 = hist3.history['acc']
epoch_curve3 = list(range(len(accuracy3)))
plt.xlabel('Epochs')
plt.ylabel('ACCURACY')
plt.plot(epoch_curve1,accuracy1,label='1 Hidden layer')
plt.plot(epoch_curve2,accuracy2,label='2 Hidden layers')
plt.plot(epoch_curve3,accuracy3,label='3 Hidden layers')
plt.legend()
plt.show()