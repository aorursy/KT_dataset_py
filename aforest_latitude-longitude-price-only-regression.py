# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_json(open("../input/train.json", "r"))

data.head()
data2 = pd.read_json(open("../input/test.json", "r"))

#data2['longitude']=data2['longitude']*(-1)

data2[['listing_id','latitude','longitude']]
x = data[['latitude','longitude','price','bathrooms','bedrooms']]

y = data[['interest_level']]



y['high'] = y['interest_level'].map({'high':1,'medium':0,'low':0}).astype(int)

y['medium'] = y['interest_level'].map({'high':0,'medium':1,'low':0}).astype(int)

y['low'] = y['interest_level'].map({'high':0,'medium':0,'low':1}).astype(int)

y = y.drop(['interest_level'], axis=1)

y.tail()
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Activation, Dropout

from keras.utils.np_utils import to_categorical


test = data2[['latitude','longitude','price','bathrooms','bedrooms']]

#test['longitude'] = test['longitude'] * (-1)



#test_predict = pd.DataFrame(test_predict) 

test
X = x.as_matrix()

Y = y.as_matrix()



xx = {}

xx[0]=data['latitude'].tolist()

xx[1]=data['latitude'].tolist()

xx[2]=data['bathrooms'].tolist()

xx[3]=data['bedrooms'].tolist()



yy=data['interest_level'].tolist





model = Sequential()



model.add(Dense(100, input_dim= 5))

model.add(Activation("relu"))

model.add(Dropout(0.5))



model.add(Dense(100))

model.add(Activation("relu"))

model.add(Dropout(0.5))



model.add(Dense(2))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



label_train_category = to_categorical(Y)

#label_test_category = to_categorical(Test)



print(label_train_category)

print(xx)



model.fit(xx, label_train_category, nb_epoch=200, batch_size=100, verbose=0)



results = model.predict_classes(test, verbose=1)
sub = pd.DataFrame()

sub['listing_id']  = data2['listing_id'].tolist()



sub['high'] = test_predict[0]

sub['medium'] = test_predict[1]

sub['low'] = test_predict[2]



sub
sub.to_csv("submission.csv", index=False)