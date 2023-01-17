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
import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import LabelEncoder





# In[239]:





train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



# Print first 5 rows



peek = train.head(20)



#print(peek)

#print(train.info())

#print(test.info())
# number of nulls

null_train = train.isnull().sum() 

#print(null)

null_test = test.isnull().sum()



#train['Sex'].hist()

count = train.groupby('Sex').size()

#print(count)



#scatter_matrix(train)
# Feature Engineering

train_test_data = [train, test]

for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

#train['Title'].value_counts()

#test['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss":1, "Mrs":2, "Master":3,"Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
# delete unnecessary feature from dataset

train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
# Titles median age for missing age

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)





# In[249]:





train.groupby("Title")["Age"].transform("median")





# In[250]:





train.info()
embarked_mapping = {"S":0, "C":1, "Q":2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
# Fill missing fare with median fare for each Pclass

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)



for dataset in train_test_data:

    dataset.loc[dataset['Fare']<=17, 'Fare'] = 0

    dataset.loc[(dataset['Fare']>17)&(dataset['Fare']<=30), 'Fare'] = 1

    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100), 'Fare'] = 2

    dataset.loc[dataset['Fare']>100, 'Fare']=3

train["Cabin"].value_counts()





# In[254]:





for dataset in train_test_data:

    dataset["Cabin"] = dataset["Cabin"].str[:1]

train["Cabin"].value_counts()





# In[255]:





cabin_mapping = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "T":7}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)





# In[256]:





train['Cabin'].fillna(train.groupby('Pclass')["Cabin"].transform("median"), inplace=True)

test['Cabin'].fillna(test.groupby('Pclass')["Cabin"].transform("median"), inplace=True)





# In[257]:





train['Cabin'].head(10)

train['Age'].value_counts()

for dataset in train_test_data:

    dataset.loc[dataset['Age']<=16, 'Age'] = 0

    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32), 'Age'] = 0.25

    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48), 'Age'] = 0.5

    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64), 'Age'] = 0.75

    dataset.loc[dataset['Age']>64, 'Age'] = 1
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
sex_mapping = {"male": 0, "female": 1}

train["Sex"] = train["Sex"].map(sex_mapping)

test["Sex"] = test["Sex"].map(sex_mapping)
features_drop = ['Ticket', 'SibSp', 'Parch']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)

id1 = test['PassengerId']

test = test.drop(['PassengerId'], axis=1)

print(id1)



# In[262]:





train_data = train.drop('Survived', axis=1)

target = train['Survived']

train_data.shape, target.shape





# In[263]:





peek = train_data.head(10)

print(train_data.shape)

#print(peek)
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
# function to create model for Keras classifier

def model():

    model = Sequential()

    model.add(Dense(12, input_dim=8, activation='relu'))

    model.add(Dense(8, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

seed = 7

np.random.seed(seed)



x = train_data

y = target



#model = KerasClassifier(build_fn=model, epochs=89, batch_size=10, verbose=0)

model = LogisticRegression()

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(model, x, y, cv=kfold)

print(results.mean())



model.fit(x, y)
print(test.head(10))

print(id1)
predictions = model.predict(test)

#print(predictions)

index = np.array(id1)

survived = np.array(predictions)

test['Survived'] = predictions

test['PassengerId'] = index

print(test.head(10))

#test[['PassengerId', 'Survived']].to_csv('C:\Radar_WS1\To_keep\00_Sources\03_Kaggle\00_Titanic\kaggle_submission.csv', index = False)

#test.to_csv('kaggle_submission5.csv', index = False)
import os

os.chdir('/kaggle/input/titanic')

#Now save your dataframe or any other file in this directory as below

#df_name.to_csv(r'df_name.csv')

a = test[['PassengerId', 'Survived']]

print(a)
#a.to_csv('kaggle_submission3.csv')

#Then in a new cell give the below command



#    from IPython.display import FileLink

#    FileLink(r'df_name.csv')