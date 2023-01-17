# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
my_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
gender_submission = pd.read_csv('../input/gender_submission.csv')
# Any results you write to the current directory are saved as output.
import pandas as pd
from sklearn.impute import SimpleImputer as Imputer
def process_data(data):
    new_data = data.reindex(sorted(data.columns), axis=1)
    #drop useless data
    #new_data=data.copy()
    useless_data = ['PassengerId','Ticket','Name','Cabin','Embarked']
    new_data=new_data.drop(labels=useless_data,axis=1)
    
    #drop category data
    category_labels = ['Sex']
    new_data = new_data.drop(labels=category_labels,axis=1)
    for label in category_labels:
        one_hot = pd.get_dummies(data[label])
        new_data = pd.concat([new_data,one_hot],axis=1)
        
    X = new_data.drop('Survived',axis=1)
    print(X.columns)
    imputer = Imputer()
    X = imputer.fit_transform(X)
    print(5)
    y = data.Survived.values
    return X,y

def process_test_data(test_data,gender_data):
    test_data['Survived'] = gender_data.Survived
    return process_data(test_data)

X,y = process_data(my_data)

X.shape
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
def train_model(my_data):
    enhanced_rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
    X_train,y_train = process_data(my_data)
    enhanced_rf.fit(X_train,y_train)
    return X,y,enhanced_rf

def test_model( test_data,gender_submission):
    X_train,y_train,enhanced_rf = train_model(my_data)
    X_test,y_test= process_test_data(test_data,gender_submission)
    return enhanced_rf.score(X_test,y_test)
    
def test_cross_eval(my_data):
    enhanced_rf = RandomForestRegressor(n_estimators = 2, random_state = 42)
    X_train,y_train = process_data(my_data)
    return cross_val_score(estimator=enhanced_rf,X=X,y=y)

def create_predictions_df(test_data):
    X_train,y_train,enhanced_rf = train_model(my_data)
    X_test,y_test= process_test_data(test_data,gender_submission)
    
    y_pred = enhanced_rf.predict(X_test)
    y_df = pd.DataFrame()
    y_df['PassengerId'] = test_data.PassengerId
    y_df['Survived'] = y_pred
    return y_df

X,y,model = train_model(my_data)
model.score(X,y)
my_submission = create_predictions_df(test_data)
my_submission.to_csv('submission2.csv', index=False) 
pd.Series(y_pred).to_csv('predictions.csv')
model.score(X,y)
score=enhanced_rf.score(X_train,y_train)
score
test_cross_eval(my_data)
test_data  = pd.read_csv('../input/test.csv')

X_test,y_test= process_test_data(test_data,gender_submission)
test_model(test_data,gender_submission)
enhanced_rd  =train_model(my_data)
