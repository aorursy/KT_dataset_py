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
!pip install pycaret
#Importing Libraries

import pandas as pd
#Reading Train and Test data

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.info()
#Dealing with missing data

train = train.drop(['Name','PassengerId','Age','Ticket','Cabin'], axis=1)
test = test.drop(['Name','PassengerId','Age','Ticket','Cabin'], axis=1)
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Fare']= test['Fare'].fillna(test['Fare'].median())
test.info()
train['Parch'].value_counts()
train['SibSp'].value_counts()
train.corr()
#Modifying the Type to object

train['Pclass'] = train['Pclass'].astype('object')
test['Pclass'] = test['Pclass'].astype('object')
#Importng pycaret and creating a classifier 

import pycaret
from pycaret.classification import *

clf = setup(data=train, target='Survived', session_id=100)
#Comparing all Classification Models

compare_models()
#Selecting XGBOOST since it has highest accuracy

model_XG = create_model('xgboost')
#Checking performance of Tuned model as well

tuned_clf = tune_model('xgboost')
#Going ahead with model_XG

plot_model(model_XG)
plot_model(model_XG, plot='pr')
evaluate_model(model_XG)
#Checking Confusion matrix on crossvalidation data

plot_model(model_XG, 'confusion_matrix')
#Prediction of Crossvalidated data
predict_model(model_XG)
#Precting on Test Data

prediction = predict_model(model_XG,test)
prediction.head()
testID = pd.read_csv('/kaggle/input/titanic/test.csv')
Sub = pd.concat([testID['PassengerId'],prediction['Label']], axis=1)
Sub.columns = [['PassengerId','Survived']]
Sub.head()
Sub.to_csv('/kaggle/working/Submission.csv', index=False)
