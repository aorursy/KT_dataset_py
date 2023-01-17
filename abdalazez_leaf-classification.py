# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/leaf-classification/train.csv.zip')

parent_data = data.copy()

ID = data.pop('id')

data
from sklearn.preprocessing import LabelEncoder

y = data['species']

y = LabelEncoder().fit(y).transform(y)

print(y.shape)

y[0:5]
from sklearn.preprocessing import StandardScaler

data.drop(columns='species',axis=1,inplace=True)

X = StandardScaler().fit(data).transform(data)

print(X.shape)
y_cat = to_categorical(y)

print(y_cat.shape)

y_cat
from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping
model = Sequential()

model.add(Dense(1500, input_dim=192,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(1300, activation='sigmoid'))

model.add(Dropout(0.1))

model.add(Dense(99, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])



early_stopping = EarlyStopping(monitor='val_loss', patience=280)

history = model.fit(X,y_cat,batch_size=192,

                    epochs=250 ,verbose=1, validation_split=0.1, callbacks=[early_stopping])
test = pd.read_csv('../input/leaf-classification/test.csv.zip')

index = test.pop('id')

test
#test_data = test

test = StandardScaler().fit(test).transform(test)

yPred = model.predict_proba(test)

yPred
#data_train = pd.read_csv('../input/leaf-classification/train.csv.zip',index_col='id')

#data_test = pd.read_csv('../input/leaf-classification/test.csv.zip')



#yPred = pd.DataFrame(yPred,index=data_test['id'],columns=sorted(data_train.species.unique()))

#data_train = pd.read_csv('../input/leaf-classification/train.csv.zip')

#test = pd.read_csv('../input/leaf-classification/test.csv.zip')

#test = StandardScaler().fit(test).transform(test)

#yPred = model.predict_proba(test)



yPred = pd.DataFrame(yPred,index=index,columns=sorted(parent_data.species.unique()))



fp = open('submission_file.csv','w')

fp.write(yPred.to_csv())

print('Done')
#yPred.to_csv('sub.csv', index=False)

#print('Done')