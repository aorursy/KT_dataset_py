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
csv_file_train = '/kaggle/input/train.csv'

csv_file_test = '/kaggle/input/test.csv'

train_data = pd.read_csv(csv_file_train)

test_data = pd.read_csv(csv_file_test)

tok1 = train_data['sentence1'].apply(lambda x: x.split(' '))

tok2 = train_data['sentence2'].apply(lambda x: x.split(' '))

test1 = test_data['sentence1'].apply(lambda x: x.split(' '))

test2 = test_data['sentence2'].apply(lambda x: x.split(' '))
from gensim.models import Word2Vec

sentences = pd.concat([tok1,tok2,test1,test2],ignore_index=True)

model_wv = Word2Vec(sentences,size=128,window=3,min_count=1,workers=1)
sentence1 = []

sentence2 = []

for x in zip(tok1,tok2):

  l1=np.array([])

  l2=np.array([])

  for k in range(17):

    if(len(x[0])>k) :

      l1 = np.append(l1,model_wv.wv[x[0][k]])

    else:

      l1=np.append(l1, np.zeros(128))

  for k in range(17):

    if(len(x[1])>k) :

      l2 = np.append(l2, model_wv.wv[x[1][k]])

    else:

      l2=np.append(l2, np.zeros(128))

  sentence1.append(l1.reshape(17,128))

  sentence2.append(l2.reshape(17,128))

    

s1=np.array(sentence1)

s2=np.array(sentence2)

s1.shape
y_train=train_data['label'].values
sentence1 = []

sentence2 = []

for x in zip(test1,test2):

  l1=np.array([])

  l2=np.array([])

  for k in range(17):

    if(len(x[0])>k) :

      l1 = np.append(l1,model_wv.wv[x[0][k]])

    else:

      l1=np.append(l1, np.zeros(128))

  for k in range(17):

    if(len(x[1])>k) :

      l2 = np.append(l2, model_wv.wv[x[1][k]])

    else:

      l2=np.append(l2, np.zeros(128))

  sentence1.append(l1.reshape(17,128))

  sentence2.append(l2.reshape(17,128))

    

t1=np.array(sentence1)

t2=np.array(sentence2)

t1.shape
from keras import Input

from keras import layers

from keras.models import Model

import keras
from keras.layers import Input,Bidirectional,LSTM,Dropout,TimeDistributed,Dense,concatenate,Flatten

sentence1 = Input(shape=(17,128),dtype='float32')

sentence2 = Input(shape=(17,128),dtype='float32')

a = Bidirectional(LSTM(64,return_sequences=True))(sentence1)

b = Bidirectional(LSTM(64,return_sequences=True))(sentence2)

# y = concatenate([a,b])

a = Dropout(0.4)(a)

b = Dropout(0.4)(b)

a = TimeDistributed(Dense(32, activation='relu'))(a)

b = TimeDistributed(Dense(32, activation='relu'))(b)

x = concatenate([a,b])

x = Dropout(0.4)(x)

x = Bidirectional(LSTM(64,return_sequences=True))(x)

x = Dropout(0.4)(x)

# output = concatenate([x,y])

x = TimeDistributed(layers.Dense(64, activation='relu'))(x)

x = Flatten()(x)

x = Dropout(0.4)(x)

label = Dense(1, activation='sigmoid')(x)

model= Model([sentence1,sentence2], label)
mcp_save = keras.callbacks.callbacks.ModelCheckpoint('model-{epoch:03d}-{val_accuracy:03f}.h5',save_best_only=False, monitor='val_loss', mode='min')

model.compile(optimizer='Adam', loss=keras.losses.binary_crossentropy,

                metrics=['accuracy'])

model.fit([s1,s2], y_train, batch_size=16,epochs=4,callbacks=[mcp_save],validation_split=0.05)
model.fit([s2,s1], y_train, batch_size=16,epochs=15,callbacks=[mcp_save],validation_split=0.05)
from keras.utils import plot_model

plot_model(model, to_file='/kaggle/working/model.png')
model.load_weights('/kaggle/working/model-003-0.813000.h5')

y_prob = model.predict([t1,t2]) 

predicted_label = (y_prob >= 0.5).astype(np.int)

test_data['label']=predicted_label
answer = pd.DataFrame(test_data,columns=['id','label'])

answer.to_csv("/kaggle/working/answer_813.csv", mode='w',index=False)
import pandas as pd

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")