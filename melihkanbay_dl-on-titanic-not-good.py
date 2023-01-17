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
import matplotlib.pyplot as plt

import re

import csv



from keras.models import Sequential

from keras.layers import Dense, Activation

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import Adam

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.layers import Dropout
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
test.head()
def preprocess(data):

    

    data.Cabin.fillna("0", inplace = True)

    

    data.loc[data.Cabin.str[0] == "A","Cabin"] = 1

    data.loc[data.Cabin.str[0] == "B","Cabin"] = 2

    data.loc[data.Cabin.str[0] == "C","Cabin"] = 3

    data.loc[data.Cabin.str[0] == "D","Cabin"] = 4

    data.loc[data.Cabin.str[0] == "E","Cabin"] = 5

    data.loc[data.Cabin.str[0] == "F","Cabin"] = 6

    data.loc[data.Cabin.str[0] == "G","Cabin"] = 7

    data.loc[data.Cabin.str[0] == "T","Cabin"] = 8

    

    data['Sex'].replace('female', 1, inplace=True)

    data['Sex'].replace('male', 2, inplace=True)

    

    data["Embarked"].replace("S",1,inplace = True)

    data["Embarked"].replace("C",2,inplace = True)

    data["Embarked"].replace("Q",3,inplace = True)

    

    data["Age"].fillna(data.Age.median(),inplace = True)

    

    data["Fare"].fillna(data.Fare.median(),inplace = True)

    

    data["Embarked"].fillna(data.Embarked.median(),inplace = True)

    

    data.drop(["PassengerId","Ticket"], axis = 1,inplace = True)

    

    return data
preprocess(train).head()
preprocess(test).head()
def group_titles(data):

    

    data['Names'] = data['Name'].map(lambda x: len(re.split(' ', x)))

    data['Title'] = data['Name'].map(lambda x: re.search(', (.+?) ', x).group(1))

    

    data['Title'].replace('Master.', 0, inplace=True)

    data['Title'].replace('Mr.', 1, inplace=True)

    data['Title'].replace(['Ms.','Mlle.', 'Miss.'], 2, inplace=True)

    data['Title'].replace(['Mme.', 'Mrs.'], 3, inplace=True)

    data['Title'].replace(['Dona.', 'Lady.', 'the Countess.', 'Capt.', 'Col.', 'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'the'], 4, inplace=True)

    

    data.drop(["Name"],axis = 1 , inplace = True)

    return data
group_titles(train).head()
train.shape
train.Cabin = train.Cabin.astype('int64') 
X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.20, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
y_train.shape
model = Sequential()

model.add(Dense(16, input_shape=X_train.shape, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

history = model.fit(X_train,y_train, epochs =300, validation_split=0.2,verbose = 1)
#model = Sequential()

#model.add(Dense(1024, input_shape=X_train.shape, activation='relu'))

#model.add(Dropout(0.2))

#model.add(Dense(1024, activation='relu'))

#model.add(Dropout(0.1))

#model.add(Dense(1024, activation='relu'))

#model.add(Dense(1, activation='sigmoid'))

#model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

#history = model.fit(X_train,y_train,batch_size = 700, epochs =10, validation_split=0.2,verbose = 1)
print(history.history.keys())
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')

plt.show()



plt.figure()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

print("accuracy: ",history.history['accuracy'][-1])

print("val_accuracy: ",history.history['val_accuracy'][-1])

print("loss: ",history.history['loss'][-1])

print("val_loss: ",history.history['val_loss'][-1])