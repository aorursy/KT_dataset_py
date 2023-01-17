import numpy as np 
import pandas as pd 
import tensorflow as tf
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path="../input/coronavirus-genome-sequence/MN908947.txt"
file=open(path,'r')
data=[[]]
for line in file:
    data.append([line])
df1=pd.DataFrame(data,columns=['sequence'])
df1
df1=df1.drop(df1.index[0],axis=0)
df1=df1.drop(df1.index[0],axis=0)
df1=df1.drop(df1.index[427],axis=0)
df1

abstracts=df1['sequence']
tokenizer=tf.keras.preprocessing.text.Tokenizer(num_words=None,filters='\n',lower=False,char_level=True)
tokenizer.fit_on_texts(abstracts)
sequences=tokenizer.texts_to_sequences(abstracts)
idx_word=tokenizer.index_word
''.join(idx_word[2] for w in sequences[10][:5])
len(idx_word)
features=[]
labels=[]
training_length=70
for seq in sequences:
    for i in range(training_length,len(seq)):
        extract=seq[i-training_length:i+1]
        features.append(extract[:-1])
        labels.append(extract[-1])
num_words=len(idx_word)+1
num_words
selector=SelectKBest(f_classif,k=10)
selected_features=selector.fit_transform(features,labels)
label_array=np.zeros((len(selected_features),num_words),dtype=np.int8)
for example_index,word_index in enumerate(labels):
    label_array[example_index,word_index]=1
label_array.shape
idx_word[np.argmax(label_array)]
sequences=np.array(sequences)
sequences=sequences.reshape(7,61,71)
label_array=label_array.reshape(7,61,6)
x_train,x_test=train_test_split(sequences,test_size=1/7,shuffle=False)
y_train,y_test=train_test_split(label_array,test_size=1/7,shuffle=False)
model=Sequential()
#recurrent layer
model.add(LSTM(16,input_shape=(x_train.shape[1:]),activation='tanh',return_sequences=True))
model.add(Dropout(0.2))
#fully connected layer
model.add(Dense(16,activation='sigmoid'))
model.add(Dropout(0.2))
#output layer
model.add(Dense(num_words,activation='softmax'))
model.summary()
opt=tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)
cce=tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=opt,loss=cce,metrics='accuracy')
history=model.fit(x_train,y_train,epochs=15,validation_data=(x_test,y_test))
