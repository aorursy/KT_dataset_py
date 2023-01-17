

import numpy as np

import pandas as pd 



data=pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')

print(data.describe())

data.head()
x=data['Message']

y=np.array(data['Category'])



#label conversion

y=np.where(y=='ham',0,1)

print(y[:6])
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



tokenizer=Tokenizer(num_words=10000)

tokenizer.fit_on_texts(x)

sequences=tokenizer.texts_to_sequences(x)

word_index=tokenizer.word_index

x=pad_sequences(sequences,maxlen=14)#first 14 words of the message

import os

glove6b='/kaggle/input/glove6b/glove.6B.100d.txt'

embeddings={}

f=open(os.path.join(glove6b))



for line in f:

    values=line.split()# list of vector representation of words

    

    word=values[0]

 

    coefs=np.asarray(values[1:],dtype='float32')#only vector representations of words

   

    embeddings[word]=coefs#dictionary containing keys and indexes

f.close

print('word count',len(embeddings))
embedding_dim=100



embedding_matrix=np.zeros((10000,embedding_dim))

for word, i in word_index.items():

    if i< 10000:#

        embedding_vector=embeddings.get(word)#word vectors

        if embedding_vector is not None:#IF WORDS FROM glove NO TO aclImdb



            embedding_matrix[i]=embedding_vector#ZERO MATRIX ASSIGNS a word from glove (vector)

           
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=1000)

print('TRAIN:',x_train.shape)

print('TEST:',x_test.shape)
from keras.models import Sequential

from keras.layers import Embedding,Flatten,Dense

from keras import layers

from keras import regularizers

from keras import optimizers

model=Sequential()

model.add(Embedding(10000,100,input_length=14))

model.add(Flatten())

model.add(Dense(64,kernel_regularizer=regularizers.l1(0.001),activation='elu'))

model.add(layers.Dropout(0.7))

model.add(Dense(1,activation='sigmoid'))
model.layers[0].set_weights([embedding_matrix]) #loading pre-trained layer

model.layers[0].trainable=False #freezing the 'embedding' layer so that it does not change its values
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history=model.fit(x_train,y_train,epochs=10,batch_size=20)

test_loss,test_acc=model.evaluate(x_test,y_test)

print('mistake:', test_loss)

print('accuracy', test_acc)