# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Flatten,Dropout
train_dir ="../input/train.csv"
raw_data = pd.read_csv(train_dir)

test_dir ="../input/test.csv"
test_data = pd.read_csv(test_dir)

#take a look
#raw_data
#data description: 16 samples each of 99 species
#990 rows * 194 collumns
col_n = pd.pivot_table(raw_data.loc[:,:'species'],columns ="species",aggfunc = np.count_nonzero)
D_label = raw_data.pop('species')
D_ID = raw_data.pop('id')
#one hot encoding
label_cat = pd.get_dummies(D_label)
#label_cat = keras.utils.to_categorical(label.values,num_classes = 99)
DN,CN = label_cat.shape
DN2,AN = raw_data.shape
model = Sequential()
model.add(Dense(128,input_dim = 192,activation ='relu'))
model.add(Dense(CN,activation ='softmax'))

model.compile(loss ='categorical_crossentropy', optimizer ='rmsprop', metrics =["accuracy"])
model.fit(raw_data,label_cat,batch_size =50, epochs =150,validation_split = 0.2)
#check overfit
model3 = Sequential()
model3.add(Dense(128,input_dim =192, activation ='relu'))
model3.add(Dropout(0.33))
model3.add(Dense(128,activation='relu'))
model3.add(Dense(CN,activation ='softmax'))

model3.compile(loss ='categorical_crossentropy', optimizer ='rmsprop',metrics =["accuracy"])
model3.fit(raw_data,label_cat,batch_size=50,epochs =100,validation_split=0.2)
model3.fit(raw_data,label_cat,batch_size=150,epochs=100,validation_split=0.2)
model4 = Sequential()
model4.add(Dense(768,input_dim =192,activation ='tanh'))
model4.add(Dropout(0.6))
model4.add(Dense(768,activation='tanh'))
model4.add(Dropout(0.6))
model4.add(Dense(99,activation ='softmax'))

model4.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics =["accuracy"])
model4.fit(raw_data,label_cat,batch_size=150,epochs =100,validation_split=0.2)
# normalize data
from sklearn import preprocessing
raw_data_nor = preprocessing.MinMaxScaler().fit_transform(raw_data)

model5=Sequential()
model5.add(Dense(360,input_dim =192,activation ='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(180,activation ='relu'))
model5.add(Dense(CN,activation ='softmax'))

model5.compile(loss = 'categorical_crossentropy',optimizer ='rmsprop',metrics =["accuracy"])
model5.fit(raw_data_nor,label_cat,batch_size=150,epochs=100,validation_split=0.2)
model5.summary()
model5.fit(raw_data_nor,label_cat,batch_size=150,epochs=100)
test_data
test_data_id = test_data.pop('id')
test_data_norm = preprocessing.MinMaxScaler().fit(raw_data).transform(test_data)
len(test_data_norm)-1
Pred = model5.predict_proba(test_data_norm,batch_size=36)
final_result = pd.DataFrame(Pred,index= test_data_id,columns=col_n.columns.values)
test_data.columns.values
fp = open('submission_leaf.csv','w')
fp.write(final_result.to_csv())
from  tensorflow.python.keras.layers  import Conv2D,Concatenate,MaxPool2D,Input,concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img,ImageDataGenerator
import matplotlib.pyplot as plt
import os,cv2
D_ID.head()




