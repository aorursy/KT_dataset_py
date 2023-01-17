# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
subm = pd.read_csv("/kaggle/input/ghouls-goblins-and-ghosts-boo/sample_submission.csv.zip")

subm.head()
train_data = pd.read_csv("/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip")

train_data.head()
test_data = pd.read_csv("/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip")

test_data.head()
train_data.info()
test_data.info()
train_data.describe()
dummies = pd.get_dummies(train_data['color'])

dummies.keys()



dummies.columns = [("color_"+i) for i in dummies.keys()]

dummies.keys()
train_data = pd.concat([train_data,dummies],axis=1)

train_data.drop("color",axis=1,inplace=True)
dummies = pd.get_dummies(test_data['color'])

dummies.columns = [("color_"+i) for i in dummies.keys()]

test_data = pd.concat([test_data,dummies],axis=1)

test_data.drop("color",axis=1,inplace=True)
train_data.head()
test_data.head()
X = train_data.drop(["id",'type'],axis=1)

X.head()
y = train_data['type']

y.head()
dummies = pd.get_dummies(y)

Y = dummies
Y.head()
from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)



print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
from keras.models import Sequential

from keras.layers import Dense,Activation



model = Sequential()



model.add(Dense(10,input_shape=(X.shape[1],)))

model.add(Activation('relu'))



model.add(Dense(10))

model.add(Activation('relu'))



model.add(Dense(3))

model.add(Activation('softmax'))



model.summary()
model.compile(optimizer='adam',

             loss='categorical_crossentropy',

             metrics=['accuracy'])
train = model.fit(X_train,Y_train,

         batch_size=8,

         epochs=50,

         verbose=2,

         validation_data=(X_test,Y_test))
plt.figure(figsize=(5,5))

plt.plot(train.history['accuracy'],'r',label='Training accuracy')

plt.plot(train.history['val_accuracy'],'b',label='Validation accuracy')

plt.legend()

plt.show()
preds = model.predict(test_data.drop("id",axis=1))

preds
preds_final = [np.argmax(i) for i in preds]

preds_final
submission = pd.DataFrame(data=test_data['id'],columns=['id'])

submission['type'] = preds_final
submission.head()
submission.replace(to_replace=[0,1,2],value=["Ghost","Ghoul","Goblin"],inplace=True)
# submission.set_index('id',inplace=True)
submission
submission.to_csv('../working/submission.csv', index=False)