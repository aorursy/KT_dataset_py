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
import pandas as pd

import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten, Activation,InputLayer

from keras.optimizers import Adam, RMSprop

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

#reading csvs and storing as data frames

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
#dropping unnecessary columns and storing passenger IDs for predictions

train_df = train_df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'])

test_Ids = test_df['PassengerId']

test_df = test_df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'])
#iterating through columns finding NAN values

for cols in train_df:

    print("col : {} -- {}= {}".format(type(train_df[cols][0]),cols,train_df[cols].isnull().sum()))
for cols in train_df:

    print("col : {} -- {}= {}".format(type(train_df[cols][0]),cols,train_df[cols].isnull().sum()))
#filling NAN values in columns

train_df = train_df.fillna(train_df['Age'].mean())

train_df = train_df.fillna(train_df['Embarked'].mode())



test_df = test_df.fillna(test_df['Age'].mean())

test_df = test_df.fillna(test_df['Fare'].mean())
#label encoder changes strings to integer representation

encoder = LabelEncoder()

train_df['Sex'] = encoder.fit_transform(train_df['Sex'].astype(str))

train_df['Embarked'] = encoder.fit_transform(train_df['Embarked'].astype(str))



test_df['Sex'] = encoder.fit_transform(test_df['Sex'].astype(str))

test_df['Embarked'] = encoder.fit_transform(test_df['Embarked'].astype(str))
#train test split to get training features, validation labels, test data and test validation labels

x_train, x_val, y_train, y_val = train_test_split(train_df.drop('Survived', axis=1), train_df['Survived'], 

                                                 test_size=0.25, random_state=42)
#Makes values passable into a neural network

train_df=pd.get_dummies(train_df)

test_df= pd.get_dummies(test_df)



train_df.head()
#Simple neural network

t_model = Sequential()



t_model.add(InputLayer(input_shape=(7,)))

t_model.add(Dense(1024, activation='relu'))

t_model.add(Dense(512, activation='relu'))

t_model.add(Dense(256, activation='relu'))

t_model.add(Dense(128, activation='relu'))

t_model.add(Dense(64, activation='relu'))



t_model.add(Dense(32, activation='relu'))

t_model.add(Dense(1, activation='sigmoid'))



t_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#training the model

t_model.fit(train_df.drop('Survived',axis=1),train_df["Survived"], epochs=100, batch_size=10,verbose=1)
preds= t_model.predict(test_df)

predictions= [0 if pred < 0.5 else 1 for pred in preds]

output = pd.DataFrame({'PassengerId': test_Ids, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")