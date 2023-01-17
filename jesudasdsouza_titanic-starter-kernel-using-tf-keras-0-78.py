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
test_path = '/kaggle/input/titanic/test.csv'

train_path = '/kaggle/input/titanic/train.csv'



train_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)
train_data.head(10)
train_data.head()

train_data.shape#(891, 12)

train_shape = train_data.shape[0]

test_shape = test_data.shape[0]
test_data.shape
train_data.isnull().sum()
train_data['Embarked'].fillna('S', inplace = True)

test_data['Embarked'].fillna('S', inplace = True)
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_Train = train_data[:train_shape][cols]

X_test = test_data[:test_shape][cols]

y = train_data[:train_shape]['Survived'].astype(int) 
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_Train = X_Train.copy()

label_X_test = X_test.copy()

Gender_col = ['Sex']

Embarked_col = ['Embarked']

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col1 in Gender_col:

    label_X_Train[col1] = label_encoder.fit_transform(X_Train[col1])

    label_X_test[col1] = label_encoder.transform(X_test[col1])

    

for col2 in Embarked_col:

    label_X_Train[col2] = label_encoder.fit_transform(X_Train[col2])

    label_X_test[col2] = label_encoder.transform(X_test[col2])    

label_X_test.head()
label_X_Train.head()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputed_label_X_Train_data = pd.DataFrame(imputer.fit_transform(label_X_Train))

imputed_label_X_test_data = pd.DataFrame(imputer.transform(label_X_test))

imputed_label_X_Train_data.columns = label_X_Train.columns

imputed_label_X_test_data.columns = label_X_test.columns
#imputed_label_X_Train_data.shape

imputed_label_X_test_data.shape
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(imputed_label_X_Train_data, y, test_size = 0.2)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

print(tf.__version__)
my_ann = Sequential()#initialising the ANN

my_ann.add(Dense(units = 4, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 7))

my_ann.add(Dense(units = 2, kernel_initializer = 'glorot_uniform', activation = 'relu'))

my_ann.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
my_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

my_ann.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose = 1)
y_pred = my_ann.predict(X_valid)

y_pred = [1 if y>=0.5 else 0 for y in y_pred]#list comprehension
from sklearn.metrics import f1_score

F1Score = f1_score(y_pred, y_valid)

print(F1Score)#f1_score of validation dataset
my_ann.fit(imputed_label_X_Train_data, y, batch_size = 10, epochs = 100, verbose = 1)
y_final = my_ann.predict(imputed_label_X_test_data)

y_final = [1 if y>=0.5 else 0 for y in y_final]
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_final })

output.to_csv("submission.csv", index = False)

print("done!")