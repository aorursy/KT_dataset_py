# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



# utils

import time

from datetime import timedelta





# Input files

file_train='../input/train.csv'

file_test='../input/test.csv'





# read training data

train_df = pd.read_csv(file_train,index_col='PassengerId')
train_df.head()
train_df.isnull().sum()
def prep_data(df):

    # Drop unwanted features

    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    

    # Fill missing data: Age and Fare with the mean, Embarked with most frequent value

    df[['Age']] = df[['Age']].fillna(value=df[['Age']].mean())

    df[['Fare']] = df[['Fare']].fillna(value=df[['Fare']].mean())

    df[['Embarked']] = df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())

    

    # Convert categorical  features into numeric

    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

      

    # Convert Embarked to one-hot

    enbarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')

    df = df.drop('Embarked', axis=1)

    df = df.join(enbarked_one_hot)



    return df
train_df = prep_data(train_df)

train_df.isnull().sum()
# X contains all columns except 'Survived'  

X = train_df.drop(['Survived'], axis=1).values.astype(float)



# It is almost always a good idea to perform some scaling of input values when using neural network models (jb).



scale = StandardScaler()

X = scale.fit_transform(X)



# y is just the 'Survived' column

y = train_df['Survived'].values
from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
print(np.shape(X))
import numpy as np

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD

from keras.utils import np_utils



model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(9,)))

model.add(Dropout(.2))

model.add(Dense(64, activation='relu'))

model.add(Dropout(.1))

model.add(Dense(1, activation='sigmoid'))





# Compiling the model

model.compile(loss = 'binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=200, verbose=2)
score = model.evaluate(X_train, y_train)

print("\n Training Accuracy:", score[1])

score = model.evaluate(X_test, y_test)

print("\n Testing Accuracy:", score[1])