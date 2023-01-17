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


#  plotting liaberies



import matplotlib

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

%matplotlib inline 



# importing ML algorithms

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# importing model validation

from sklearn.model_selection import train_test_split



# importing model evaluation matrix

from sklearn.metrics import confusion_matrix
df_titanic_train = pd.read_csv('/kaggle/input/train.csv')

print('shape of train data ',df_titanic_train.shape )
#getting null values 

NAN_train = pd.DataFrame(df_titanic_train.isnull().sum(),columns=['Train'])

NAN_train



# getting only null columns 

NAN_train[NAN_train['Train']>1]
# we will drop the Cabin feature since it is missing a lot of the data

# we are also dropping some unwanted data 



df_titanic_train.pop('Cabin')

df_titanic_train.pop('Name')

df_titanic_train.pop('Ticket')

df_titanic_train.pop('PassengerId')

df_titanic_train.pop('Fare')

df_titanic_train.shape
# # Filling missing Age values with mean

df_titanic_train['Age'] = df_titanic_train['Age'].fillna(df_titanic_train['Age'].mean())

# # Filling missing Embarked values with most common value

df_titanic_train['Embarked'] = df_titanic_train['Embarked'].fillna(df_titanic_train['Embarked'].mode()[0])

df_titanic_train.dtypes
# # Getting Dummies from all other categorical vars

for col in df_titanic_train.dtypes[df_titanic_train.dtypes == 'object'].index:

    for_dummy = df_titanic_train.pop(col)

    df_titanic_train = pd.concat([df_titanic_train, pd.get_dummies(for_dummy, prefix=col)], axis=1)
labels = df_titanic_train.pop('Survived')
x_train, x_test, y_train, y_test = train_test_split(df_titanic_train, 

                                                    labels, 

                                                    test_size=0.25,

                                                    random_state=4)
#model = KNeighborsClassifier()

model = RandomForestClassifier(random_state=60)

model.fit(x_train, y_train)
# Predict for all the test instances

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred) 

cm
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred)

score