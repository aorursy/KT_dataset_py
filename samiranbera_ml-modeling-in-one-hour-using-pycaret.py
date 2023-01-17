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
from pycaret.classification import *
train = pd.read_csv('../input/titanic/train.csv')
test  = pd.read_csv('../input/titanic/test.csv')
sub   = pd.read_csv('../input/titanic/gender_submission.csv')

print('Training Data = ',train.shape)
print('Testing Data = ',test.shape)
train.head(3)
train_copy = train.copy()
test_copy = test.copy()
train = train_copy.copy()
test  = test_copy.copy()

# Get Title and Family Name
train['title']     = train.Name.apply(lambda x: x.split(',')[1][1:].split()[0][:-1])
train['last_name'] = train.Name.apply(lambda x: x.split(',')[0][:-1])
test['title']      =  test.Name.apply(lambda x: x.split(',')[1][1:].split()[0][:-1])
test['last_name']  =  test.Name.apply(lambda x: x.split(',')[0][:-1])

# Impute Missing Values of Age
train.Age[train.Age.isna()] = train.Age.mean()
test.Age[test.Age.isna()]   = test.Age.mean()

# Transform Cabin
train.Cabin = train.Cabin.replace(np.nan, 'N0', regex=True)
train['Cabin_Cat'] = [i[0]  for i in train.Cabin]
test.Cabin  = test.Cabin.replace(np.nan, 'N0', regex=True)
test['Cabin_Cat']  = [i[0]  for i in test.Cabin]

#Drop Rows with Missing Values - Embarked
train = train[~train.Embarked.isna()]
test  = test[~test.Embarked.isna()]

# Drop Column
train.drop(columns = ['Ticket','Cabin','Name'],axis=0, inplace=True)
test.drop(columns  = ['Ticket','Cabin','Name'],axis=0, inplace=True)

train.head(3)
clf1 = setup(data                 = train, 
             target               = 'Survived',                                      
             ignore_features      = ['PassengerId'],
             categorical_features = ['Pclass','Sex','Parch','Embarked','title','Cabin_Cat'],
             numeric_features     = ['Age','SibSp','Fare'],                               
             session_id           = 123)

compare_models(sort='AUC',fold=10)
lr = create_model('lr',fold=10)
tune_lr = tune_model('lr',fold=10,optimize='Accuracy')
plot_model(tune_lr, plot = 'auc')
plot_model(tune_lr,plot='confusion_matrix')
plot_model(tune_lr, plot='feature')
xgboost = create_model('xgboost',fold=10)
#tune_xgb = tune_model('xgboost',optimize = 'Accuracy',fold=10)
plot_model(xgboost, plot = 'auc')
plot_model(xgboost, plot = 'confusion_matrix')
plot_model(xgboost, plot = 'feature')
interpret_model(xgboost)
lgb = create_model('lightgbm',fold=10)
#tune_lgb = tune_model('lightgbm',optimize = 'Accuracy',fold=10)
plot_model(lgb, plot='auc')
plot_model(lgb, plot='confusion_matrix')
plot_model(lgb, plot='feature')
dt = create_model('dt', fold  =10)
dt_bagging = ensemble_model(dt, method='Bagging', fold=10)
plot_model(dt_bagging, plot='auc')
plot_model(dt_bagging, plot='confusion_matrix')
dt = create_model('dt',fold=10)
dt_boosting = ensemble_model(dt, method='Boosting')
plot_model(dt_boosting,plot='auc')
plot_model(dt_boosting,plot='confusion_matrix')
xgb_pred = predict_model(xgboost, data=test)
xgb_pred.head(3)
sub['Survived'] = round(xgb_pred['Score']).astype(int)
sub.to_csv('submission.csv',index=False)
sub.head(10)