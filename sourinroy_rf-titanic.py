# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Reading the train and test data



train_data = pd.read_csv('../input/titanic/train.csv',index_col = 'PassengerId')

test_data = pd.read_csv('../input/titanic/test.csv',index_col = 'PassengerId')



#Dropping columns which seem to be un-important



train_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

test_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)



#Filling NaN values and replacing categorical values with numeric data



train_data['Sex'].replace({'male': 1 , 'female' : 0} ,inplace = True)

train_data['Embarked'].replace( {'S' : 2 , 'C' : 1 , 'Q' : 0 } , inplace = True )

train_data['Age'].fillna( (train_data['Age'].mean()) , inplace = True) 

train_data['Embarked'].fillna(3,inplace=True)



test_data['Sex'].replace({'male': 1 , 'female' : 0} ,inplace = True)

test_data['Embarked'].replace( {'S' : 2 , 'C' : 1 , 'Q' : 0 } , inplace = True )

test_data['Age'].fillna((test_data['Age'].mean()), inplace = True)

test_data['Embarked'].fillna(3,inplace=True)

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)



#train_data.isnull().sum()

#defining features and target



features = ['Pclass', 'Sex', 'SibSp', 'Fare', 'Age','Parch','Embarked']

X = train_data[features]

y = train_data.Survived

#splitting of data into training and testing data



from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=11,test_size=0.25)
#Using Random Forest Classification for fitting data and prediction



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



Rf_model = RandomForestClassifier(n_estimators=1000,max_depth=15,random_state=3)

Rf_model.fit(train_X,train_y.values.ravel())

preds = Rf_model.predict(val_X)

mean_absolute_error(preds,val_y)
#Fitting the model to test data



FullRf_model = RandomForestClassifier(n_estimators=1000,max_depth=15,random_state=3)

FullRf_model.fit(X,y)

final_pred = FullRf_model.predict(test_data)
#Exporting as .csv 



output = pd.DataFrame({'PassengerId': test_data.index, 'Survived': final_pred})

output.to_csv('my_submission.csv', index=False)

print("Saved!")