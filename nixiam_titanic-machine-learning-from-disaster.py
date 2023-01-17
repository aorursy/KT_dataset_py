import matplotlib.pyplot as plt
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_result = pd.concat([df_train, df_test], ignore_index=True, sort=False)

print(f"{df_train.shape} - {df_test.shape} - {df_result.shape}")
df_result.head(20)
df_result.tail(20)
df_result.columns
missing_v = df_result.isnull().sum().sort_values(ascending=False)

missing_v[missing_v > 0]
df_result['Cabin'].value_counts(dropna=False)
print("-- convert cabin to class 1 where cabin 0 where not cabin --")

df_result['Cabin'].fillna("0", inplace=True)

df_result['Cabin'] = df_result['Cabin'].apply(lambda x: "1" if x != "0" else "0")

df_result['Cabin'].value_counts(dropna=False)
df_result.groupby('Pclass')['Age'].describe()

df_result['Age'] = df_result['Age'].fillna(df_result.groupby('Pclass')['Age'].transform('mean'))
df_result['Age'] = pd.qcut(df_result['Age'].rank(method='first'), 10, labels=[1, 2, 3, 4, 5, 6, 7 , 8 , 9 , 10]).astype(str)

df_result['Age'].value_counts()
df_result['Embarked'].value_counts(dropna=False)
df_result.groupby(['Pclass','Cabin'])['Embarked'].value_counts(dropna=False)
df_result['Embarked'].fillna("S", inplace=True)

df_result['Embarked'].value_counts(dropna=False)
df_result['Embarked'] = df_result['Embarked'].map({"S":"0", "C":"1", "Q":"2"})
df_result.groupby(["Pclass",'Cabin','Embarked'])['Fare'].describe()
df_result['Fare'] = df_result['Fare'].fillna(df_result.groupby(["Pclass",'Cabin','Embarked'])['Fare'].transform('mean'))
fare_min = df_result['Fare'].min()

fare_max = df_result['Fare'].max()

print(fare_min, fare_max)
df_result['Fare'] = pd.qcut(df_result['Fare'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(str)

df_result['Fare'].value_counts()
missing_v = df_result.isnull().sum().sort_values(ascending=False)

missing_v[missing_v > 0]
df_result.head()
df_result = df_result.drop(['Name','Ticket'], axis="columns")
df_result.head()
df_result['Sex'] = df_result['Sex'].apply(lambda x: "1" if x == "male" else "0")

df_result['Sex'].head()
df_result.head()
df_result = df_result.drop(['PassengerId'], axis=1)

df_result.head()
df_result.info()
df_result['Survived'] = df_result['Survived'].astype(str)

df_result['Pclass'] = df_result['Pclass'].astype(str)
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense
# -*- coding: utf-8 -*-

"""

Created on Mon Feb 17 11:29:16 2020



@author: benedet

"""



import numpy as np

import pandas as pd

from sklearn.base import TransformerMixin

from sklearn.preprocessing import OneHotEncoder

class DataFrameEncoder(TransformerMixin):



    def __init__(self):

        """Encode the data.



        Columns of data type object are appended in the list. After 

        appending Each Column of type object are taken dummies and 

        successively removed and two Dataframes are concated again.



        """

    def fit(self, X, y=None):

        self.object_col = []

        self.hotencoders_list = {}

        for col in X.columns:

            #print(col, X[col].dtype)

            if(X[col].dtype == np.dtype('O') or X[col].dtype == np.dtype('str')):

                self.object_col.append(col)

                colum = np.array(X[col]).reshape(-1, 1)

                #print(colum, colum.dtype)

                self.hotencoders_list[col] = OneHotEncoder(handle_unknown='ignore').fit(colum)

        return self



    def transform(self, X, y=None, drop = True):

        for col in self.object_col:

            #print(col)

            colum = np.array(X[col]).reshape(-1, 1)

            #print(colum, self.hotencoders_list[col].categories_)

            tmp = self.hotencoders_list[col].transform(colum).toarray()

            if drop:

                tmp = pd.DataFrame(tmp[:, 1:])

            else:

                tmp = pd.DataFrame(tmp)

            X = X.drop(col,axis=1)

            X = pd.concat([tmp,X],axis=1)

        return X

    

    def inverse_transform(self, X, y=None, drop = True):

        out = pd.DataFrame(columns=self.object_col)

        counter = 0

        for col in self.object_col:

            cat = len(self.hotencoders_list[col].categories_) + 1

            tmp = self.hotencoders_list[col].inverse_transform(X[counter:cat])

            tmp = pd.DataFrame(tmp)

            out = pd.concat([tmp,out],axis=1)

            counter = counter + 1

        return out
X_tot = df_result.loc[:, df_result.columns != 'Survived']

Y_tot = df_result.loc[:, 'Survived'].values

Y_tot = Y_tot.reshape(-1, 1)

print(f"X_tot shape:{X_tot.shape} - Y_tot shape:{Y_tot.shape}")
de = DataFrameEncoder().fit(X_tot)

X_tot = de.transform(X_tot)

print(f"X_tot shape:{X_tot.shape}")
X = X_tot[:891]

Y = Y_tot[:891]

print(f"X shape:{X.shape} - Y shape:{Y.shape}")
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

scaler_X = MinMaxScaler(feature_range=(0, 1))

scaler_y = MinMaxScaler(feature_range=(0, 1))



X = scaler_X.fit_transform(X)

Y = scaler_y.fit_transform(Y)
print("-- Splitting the dataset into the Training set and Val set --")

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.1, random_state = 42)



#X_train = X

#y_train = Y



print(f"X_train shape:{X_train.shape} - y_train shape:{y_train.shape}")

print(f"X_val shape:{X_val.shape} - y_val shape:{y_val.shape}")
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Dropout

from tensorflow.keras.optimizers import Adam



lr = 0.0005

adam = Adam(lr)



input_data = Input(shape=(X_train.shape[1],))



x = Dense(name="Dense_1", units=32, activation='relu')(input_data)

x = Dense(name="Dense_2", units=32, activation='relu')(x)

o = Dense(units=y_train.shape[1], activation='sigmoid')(x)



model = Model(inputs=[input_data], outputs=[o])

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), verbose=1)
def visualize_learning_curve(history):

    # list all data in history

    print(history.history.keys())

    # summarize history for accuracy

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model mse')

    plt.ylabel('mse')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



visualize_learning_curve(history)
X_test = X_tot[891:]

print(X_test.shape)
Y_pred = model.predict(X_test)

Y_pred = np.round(Y_pred, decimals=0).astype(int)

print(Y_pred.shape)
df_sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

df_sub.shape
df_sub.head()
df_sub['Survived'] = Y_pred
df_sub.head()
df_sub.to_csv("submission.csv", index=False)