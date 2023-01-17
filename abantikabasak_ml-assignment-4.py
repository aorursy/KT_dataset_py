# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re 
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
import bz2
bz_file=bz2.BZ2File("../input/test.ft.txt.bz2")
line_list=bz_file.readlines()
#print(line_list)
print(len(line_list))
print(line_list[0])#2 indicates good, 1 indicates bad
print(line_list[2])

def Frame_data(listt):
    Lst=[]
    for line in listt:
        #print(line)
        #print(type(line))
       
        line=str(line)
        txtline=line.split('__label__')[1] #Choose the Right hand side of __label__
        #print("Textline"+str(txtline))
        txtline=re.sub('[!?:;,.|''""]','',txtline) #Substitute these characters with ''
        #print("Textline"+str(txtline))
        temp=txtline.split(' ') #Split at splaces
        #print("Temp"+str(temp))
        sentiment="good" if(temp[0]=="2") else "bad"
        sentence=temp[1:len(temp)]
        #print("Sentence:\n"+str(sentence))
        sentence[len(sentence)-1]=sentence[len(sentence)-1][0:len(sentence[len(sentence)-1])-3]
        #print("Sentence:\n"+str(sentence))
        sentence=' '.join(sentence)
        #print("Sentence:\n"+str(sentence))
        Lst.append([sentiment,sentence])
        
    DFrame=pd.DataFrame(Lst,columns=("Sentiment_class_label","Review_Text"))
    DFrame.to_csv("Sentiment.csv")
Frame_data(line_list)





InputFile="Sentiment.csv"
DFrame=pd.read_csv(InputFile,index_col=0)
print(DFrame.head(10))
num_rows=len(DFrame)
word_length=[]
for i in range(0,num_rows):
    word_length.append(len(DFrame["Review_Text"][i].split(' ')))
print(word_length)
DFrame["Word_Length"]=word_length
print(DFrame.head(21))



pos=np.where(DFrame["Word_Length"]<25)
print(pos[0])
#Replacing good with 1 and bad with 0

DFrame2=DFrame.loc[pos[0]]
DFrame2["Sentiment_class_label"][DFrame2["Sentiment_class_label"]=="good"]=1
DFrame2["Sentiment_class_label"][DFrame2["Sentiment_class_label"]=="bad"]=0
print(DFrame2.head(10))
num_rows=len(DFrame2)
num_cols=len(DFrame2.columns)
print("Dimensions of new dataframe = ("+str(num_rows)+","+str(num_cols)+")")

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
stopword = text.ENGLISH_STOP_WORDS
vector = CountVectorizer(stop_words = stopword,min_df=.0001,lowercase=1) #Builds vocabulary 
X = vector.fit_transform(DFrame2['Review_Text'].values)
Words=vector.get_feature_names()

print("Number of words = "+str(len(Words)))
print(Words)

print(X.toarray())
print(len(X.toarray()))
print(len(X.toarray()[0]))
from sklearn.model_selection import train_test_split
y = DFrame2['Sentiment_class_label'].values
train_x, test_x, train_y, test_y = train_test_split(X.toarray(), y, test_size=0.1, random_state=1)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
import tensorflow as tf
import keras
    
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

mod= Sequential()
mod.add(Dense(1000,input_shape=(train_x.shape[1],),activation='relu'))
mod.add(Dense(1,activation='sigmoid'))

mod.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
TRAIN = mod.fit(train_x,train_y,epochs=5,batch_size=128,verbose=1)
Y1=mod.predict(train_x)
Y1[Y1>=0.5]=1
Y1[Y1<0.5]=0
Y1=Y1.astype("int")
Y1=Y1.reshape(train_y.shape)
acc=sum((Y1==train_y)*1)*1.0/Y1.shape[0]
print ("Accuracy = "+str(acc))
Y1=mod.predict(test_x)
Y1[Y1>=0.5]=1
Y1[Y1<0.5]=0
Y1=Y1.astype("int")
Y1=Y1.reshape(test_y.shape)
acc=sum((Y1==test_y)*1)*1.0/Y1.shape[0]
print ("Accuracy = "+str(acc))
import matplotlib.pyplot as plt
loss_curve = TRAIN.history['loss']
epoch = list(range(len(loss_curve)))
plt.plot(epoch,loss_curve)
mod2= Sequential()
mod2.add(Dense(1000,input_shape=(train_x.shape[1],),activation='relu'))
mod2.add(Dense(1000,activation='relu'))
mod2.add(Dense(1,activation='sigmoid'))

mod2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
TRAIN = mod2.fit(train_x,train_y,epochs=5,batch_size=128,verbose=1)