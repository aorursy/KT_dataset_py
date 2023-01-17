# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Apply Light GBM
import os
import re
import numpy as np
import pandas as pd
from sklearn import tree  
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
def preprocessingData(df, type = "train"):
    # Replace string fields by number
    df['Sex'] = df['Sex'].replace('male', 1)
    df['Sex'] = df['Sex'].replace('female', 0)

    # Add three more features "FamilySize" and "Master"
    # FamilySize = SibSp + Parch.
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # Minor = 1 if 'Master' title appears in name and Minor = 0 if 'Master' does not appear in name. 
    df['Minor'] = 0
    for i in range(len(df.Name)):
        if "Master" in df.Name[i]:
            df.Minor[i] = 1 

    df['Surname'] =  df.Name.str.extract("([A-Z]\w{0,})")
    
    if(type == "train"):
        df['FamilyOneSurvived'] = 0
        df['FamilyAllDied'] = 0
        for i in range(len(df.Surname)):
            for j in range(i+1, len(df.Surname)):
                if df.Surname[i] == df.Surname[j] and (df.Survived[i] == 1 or df.Survived[j] == 1):
                    df.FamilyOneSurvived[i] = 1
                    df.FamilyOneSurvived[j] = 1

        for i in range(len(df.Surname)):
            for j in range(i+1, len(df.Surname)):
                if df.Surname[i] == df.Surname[j] and df.FamilyOneSurvived[i] == 0:
                    df.FamilyAllDied[i] = 1
                    df.FamilyAllDied[j] = 1

    # Drop unnecessary features
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked'], axis =1)
    return df
def addTwoFeatures(df1, df2):
    df2['FamilyOneSurvived'] = 0
    df2['FamilyAllDied'] = 0
    for i in range(len(df2.Surname)):
        for j in range(len(df1.Surname)):
            if df2.Surname[i] == df1.Surname[j]:
                if df1.FamilyOneSurvived[j] == 1:
                    df2.FamilyOneSurvived[i] = 1

                if df1.FamilyAllDied[j] == 1:
                    df2.FamilyAllDied[i] = 1 
    return df2
# Get Passenger ID in test data 
# It will be used to add into final result
id = test['PassengerId']
id_df = pd.DataFrame(id)
# Preprocessing data
train_df = preprocessingData(train)
test_df = preprocessingData(test, "test")
test_df = addTwoFeatures(train_df, test_df)

train_df = train_df.drop(['Surname'], axis = 1)
test_df = test_df.drop(['Surname'], axis = 1)
# Init train and validate data from training data set
train, validate = np.split(train_df.sample(frac=1), [int(.7*len(train_df))])
y_train = train['Survived']
x_train = train.drop(['Survived'], axis = 1)

y_validate = validate['Survived']
x_validate = validate.drop(['Survived'], axis = 1)
d_train = lgb.Dataset(x_train, label= y_train)

params = {}
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['max_depth'] = 3
clf = lgb.train(params, d_train, 100)
#Prediction
y_valid = clf.predict(x_validate)
#convert into binary values
for i in range(len(y_valid)):
    if y_valid[i]>=.5:       # setting threshold to .5
        y_valid[i]= 1
    else:  
        y_valid[i]= 0

accuracy=accuracy_score(y_valid,y_validate)
print(accuracy)
y_predicted = clf.predict(test_df)
for i in range(len(y_predicted)):
    if y_predicted[i]>=.5:       # setting threshold to .5
        y_predicted[i]= 1
    else:  
        y_predicted[i]= 0
predicted = pd.DataFrame({'Survived': y_predicted})
predicted = predicted.astype(int) 
# Join predicted into result dataframe and write result as a CSV file
result = id_df.join(predicted)
result.to_csv("result_final.csv", index = False)