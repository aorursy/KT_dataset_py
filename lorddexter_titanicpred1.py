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
Train = pd.read_csv('/kaggle/input/titanic/train.csv')

Test = pd.read_csv('/kaggle/input/titanic/test.csv')

# Checking for missing values

Train.isnull().sum()



# Filling up missing values in the Age and Embarked column

Train['Age'].fillna(method = 'ffill', inplace = True)

Train['Embarked'] = Train['Embarked'].fillna('S')



# Name, Ticket, Cabin and Embarkation point have no impact so deleted

Train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)



# Hard-coding the categorical columns to category type for easier handling

Train['Pclass'] = Train['Pclass'].astype('category')

Train['Sex'] = Train['Sex'].astype('category')

Train['Embarked'] = Train['Embarked'].astype('category')



Train.dtypes



# Fixing the categorical variables

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

Train['Pclass'] = label_encoder.fit_transform(Train['Pclass'])

Train['Sex'] = label_encoder.fit_transform(Train['Sex'])

Train['Embarked'] = label_encoder.fit_transform(Train['Embarked'])

Train = pd.get_dummies(Train, columns = ['Pclass'], prefix = ['Pclass'])

Train = pd.get_dummies(Train, columns = ['Sex'], prefix = ['Sex'])

Train = pd.get_dummies(Train, columns = ['Embarked'], prefix = ['Embarked'])



X = Train.iloc[:, 1:13].values

Y = Train.iloc[:, 0].values



# Splitting training set

# from sklearn.model_selection import train_test_split

# Train_X, Val_X, Train_Y, Val_Y = train_test_split(X, Y, test_size = 0.2)



# Setting up the ANN

import keras

from keras.models import Sequential

from keras.layers import Dense



model = Sequential([Dense(units = 6, activation = 'relu', input_shape = (12, )), 

                    Dense(units = 6, activation = 'relu'),

                    Dense(units = 1, activation = 'sigmoid'),

                    ])



model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



model.fit(X, Y, epochs = 100)



Test.isnull().sum()



Test['Age'].fillna(method = 'ffill', inplace = True)

Test['Fare'].fillna(method = 'ffill', inplace = True)



Test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)



Test['Pclass'] = Test['Pclass'].astype('category')

Test['Sex'] = Test['Sex'].astype('category')

Test['Embarked'] = Test['Embarked'].astype('category')



from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

Test['Pclass'] = label_encoder.fit_transform(Test['Pclass'])

Test['Sex'] = label_encoder.fit_transform(Test['Sex'])

Test['Embarked'] = label_encoder.fit_transform(Test['Embarked'])

Test = pd.get_dummies(Test, columns = ['Pclass'], prefix = ['Pclass'])

Test = pd.get_dummies(Test, columns = ['Sex'], prefix = ['Sex'])

Test = pd.get_dummies(Test, columns = ['Embarked'], prefix = ['Embarked'])



y_pred = model.predict(Test)

y_pred = np.rint(y_pred)
