# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def contain(a,b):

    for i in a:

        if i in b:

            return False

    return True

train_raw_data=json.load(open('../input/train.json','r'))

raw_target_class={}

target_count=0

raw_ingredients={}

trans={}

ingredients_count=0

for i in train_raw_data:

    if i['cuisine'] not in raw_target_class:

        raw_target_class[i['cuisine']]=target_count

        trans[target_count]=i['cuisine']

        target_count=target_count+1

    for j in i['ingredients']:

        for k in j.lower().split():

            if k not in raw_ingredients and contain(raw_ingredients,k):

                raw_ingredients[k]=ingredients_count

                ingredients_count=ingredients_count+1



target_data=np.zeros((len(train_raw_data),len(raw_target_class)))

train_data=np.zeros((len(train_raw_data),len(raw_ingredients)))

num=0

for i in train_raw_data:

    target_data[num][raw_target_class[i['cuisine']]]=1

    for j in i['ingredients']:

        for k in j.lower().split():

            for l in raw_ingredients:

                if l in k:

                    train_data[num][raw_ingredients[l]]=1

    num=num+1

    

test_raw_data=json.load(open('../input/test.json','r'))

test_data=np.zeros((len(test_raw_data),len(raw_ingredients)))

test_id=[]

num=0

for i in test_raw_data:

    test_id.append(i['id'])

    for j in i['ingredients']:

        for k in j.split():

            for l in raw_ingredients:

                if l in k:

                    test_data[num][raw_ingredients[l]]=1

    num=num+1



print(len(train_raw_data))

print(len(raw_ingredients))



# Any results you write to the current directory are saved as output.
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import backend as K



batch_size = 1000

epochs = 300



model = Sequential()

model.add(Dense(1000,activation='tanh',input_shape=(len(raw_ingredients),)))

model.add(Dropout(0.25))

model.add(Dense(200,activation='tanh'))

model.add(Dense(100,activation='tanh'))



model.add(Dense(50,activation='tanh'))



model.add(Dropout(0.25))

model.add(Dense(len(raw_target_class),activation='softmax'))

model.compile(loss=keras.losses.mean_squared_error,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.fit(train_data, target_data,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1)

raw_result=model.predict(test_data)

def find_max(a):

    i=0

    j=0

    for k in range(len(a)):

        if a[k]>j:

            i=k

            j=a[k]

    return i

result=[]

for i in range(len(raw_result)):

    result.append(trans[find_max(raw_result[i])])

pd.DataFrame.from_dict({'id':test_id,'cuisine':result}).to_csv('submission.csv', index = False)

print('done')