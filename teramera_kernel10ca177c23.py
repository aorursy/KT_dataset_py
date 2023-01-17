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
tr=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip')

tr.head()
ts=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip')

ts.head()
tr.shape
ts.shape
tr.isnull().sum()
ts.isnull().sum()
tr['bone_length'].value_counts()
tr['rotting_flesh'].value_counts()
tr['color'].value_counts()
train_data=pd.concat([tr.drop('color',axis=1),pd.get_dummies(tr['color'])],axis=1)
train_data.head()
test_data=pd.concat([ts.drop('color',axis=1),pd.get_dummies(ts['color'])],axis=1)
test_data.head()
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(train_data.drop(['id','type'],axis=1),pd.get_dummies(train_data['type']))
xts.shape
yts.shape
from sklearn.neural_network import MLPClassifier
cl=MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=(100,10),random_state=10,activation='logistic',max_iter=2000)
cl.fit(xtr,ytr)
cl.score(xts,yts)
df=cl.predict(xts)
pred=cl.predict(test_data.drop('id',axis=1))
from tensorflow import keras
from keras.layers import Dense,Activation

from keras.models import Sequential
model=Sequential()

model.add(Dense(10,input_shape=(10,)))

model.add(Activation('relu'))

model.add(Dense(10))

model.add(Activation('relu'))

model.add(Dense(3))

model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(xtr,ytr,validation_data=(xts,yts),verbose=1,batch_size=10,epochs=100)
pred=model.predict(test_data.drop('id',axis=1))
pred
pred_final=[np.argmax(i) for i in pred]
pred
submission = pd.DataFrame({'id':test_data['id'], 'type':pred_final})

submission.head()
submission['type'].replace(to_replace=[0,1,2],value=['Ghost','Ghoul','Goblin'],inplace=True)

submission.head()
submission.to_csv('../working/submission.csv', index=False)