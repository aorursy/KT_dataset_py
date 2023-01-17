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
import tensorflow as tf
from zipfile import ZipFile

with ZipFile('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip', 'r') as test:

    test.extractall()



    
with ZipFile('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip', 'r') as train:

    train.extractall()

with ZipFile('/kaggle/input/ghouls-goblins-and-ghosts-boo/sample_submission.csv.zip', 'r') as sample_submission:

    sample_submission.extractall()
sample = pd.read_csv("sample_submission.csv")
sample.head(25)
test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")
test_data.head(25)
train_data.info()
print(test_data.shape)

print(train_data.shape)
train_data.head()
train =train_data.drop(["id",'color','type'], axis=1)

train
test = test_data.drop(["id",'color'], axis=1)

test
categorical=train_data.columns[train_data.dtypes=='object']

categorical
dummies=pd.get_dummies(train_data['color'])

dummies.columns=str('color')+'_'+dummies.columns

train=pd.concat([train,dummies],axis=1)

train
dummies=pd.get_dummies(test_data['color'])

dummies.columns=str('color')+'_'+dummies.columns

test=pd.concat([test,dummies],axis=1)

test
X_train = np.array(train)

X_train.shape
X_test = np.array(test)

X_test.shape
pd.get_dummies(train_data['type'])
Y_train = np.array(pd.get_dummies(train_data['type']))
Y_train
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten
model = Sequential()



model.add(Dense(100, input_shape=(10,)))

model.add(Activation('relu'))



model.add(Dense(100))

model.add(Activation('relu'))



model.add(Dense(100))

model.add(Activation('relu'))



model.add(Dense(3))

model.add(Activation('softmax'))



model.summary()
model.compile(loss='categorical_crossentropy',

             optimizer = 'adam',

             metrics = ['accuracy'])
model.fit(X_train,Y_train,

         batch_size=12,

         epochs = 32,

         verbose=2)
pred = model.predict(X_test).argmax(axis =1)
Y_train
submit = pd.DataFrame({'id':test_data['id'],

                      'type': pred})

submit
submit.replace(0,'Ghost')
submit.replace(1,'Ghoul')

submit.replace(2,'Goblin')
submission=submit.replace(0,'Ghost')
submission=submission.replace(1,'Ghoul')

submission=submission.replace(2,'Goblin')

submission
submission.to_csv('submission.csv',index = False)