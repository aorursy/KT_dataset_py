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
#Read csv files to Dataframes

train_data= pd.read_csv('../input/titanic/train.csv')

test_data= pd.read_csv('../input/titanic/test.csv')



# DATA CLEANING

#print(train_data.head())



train_data.loc[(train_data['PassengerId'].isnull())|(train_data['Survived'].isnull())|(train_data['Pclass'].isnull())|

              (train_data['Name'].isnull())|(train_data['Sex'].isnull())|(train_data['Age'].isnull())|

               (train_data['SibSp'].isnull())|(train_data['Parch'].isnull())|(train_data['Ticket'].isnull())|

              (train_data['Ticket'].isnull())|(train_data['Fare'].isnull())|(train_data['Cabin'].isnull())|

              (train_data['Embarked'].isnull())]
#dropping rows with missiing values

train_data=train_data.dropna()

#coverting gender data to numerical data. 

train_data=train_data.replace(to_replace=['female','male'], value=[1,0])

train_data
#convert numerical values into numpy array to be used in M.L models



# input features for training data

all_training_features=train_data[['Pclass','Sex','Age']].values

# output results for training data

all_training_classes=train_data[['Survived']].values
#Normalizing training data

from sklearn import preprocessing



scaler = preprocessing.StandardScaler()

all_training_features_scaled = scaler.fit_transform(all_training_features)

all_training_features_scaled
# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

LR=LogisticRegression()

scores=cross_val_score(LR,all_training_features_scaled,all_training_classes,cv=3)

scores.mean()
#test_data.head()
# DATA CLEANING FOR TEST DATA

#Dropping rows with missing data

test_data=test_data.dropna()

#coverting gender data to numerical data. 

test_data=test_data.replace(to_replace=['female','male'], value=[1,0])

test_data.head()
# input features from the test data

all_testing_features=test_data[['Pclass','Sex','Age']].values



#Normalizing test data



scaler2 = preprocessing.StandardScaler()

all_testing_features_scaled = scaler2.fit_transform(all_testing_features)

all_testing_features_scaled
#Fitting the Logistic regression model for predictions

LR.fit(all_training_features,all_training_classes)



#predictions

y_predict=LR.predict(all_testing_features_scaled)

y_predict
test_data.insert(1,'Survived',y_predict,True)

test_data
# A new Dataframe for exporting to csv file

test_data=test_data[['PassengerId','Survived']]

test_data
# Exporting to csv file

test_data.to_csv('Predictions.csv',index=False)