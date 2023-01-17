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
#import necessary modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#download dataset
train=pd.read_csv("../input/titanic/train.csv")
test=pd.read_csv("../input/titanic/test.csv")

#split train data
train_label=train["Survived"]
train=train.drop("Survived", axis=1)
#check data
train.head()
#check lacking data
def lack_table(df):
    lack_number=df.isnull().sum()
    lack_ratio=lack_number/len(df)
    lack_table=pd.concat([lack_number, lack_ratio], axis=1)
    lack_table=lack_table.rename(columns={0: "lack_number", 1: "lack_ratio"})
    return lack_table

lack_table(train)
lack_table(test)
#convert categorical data to number_label
train["Sex"][train["Sex"]=="male"]=0
train["Sex"][train["Sex"]=="female"]=1
train["Embarked"][train["Embarked"]=="S"]=0
train["Embarked"][train["Embarked"]=="C"]=1
train["Embarked"][train["Embarked"]=="Q"]=2

test["Sex"][test["Sex"]=="male"]=0
test["Sex"][test["Sex"]=="female"]=1
test["Embarked"][test["Embarked"]=="S"]=0
test["Embarked"][test["Embarked"]=="C"]=1
test["Embarked"][test["Embarked"]=="Q"]=2
#compensate lack data
train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)

train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test["Embarked"].fillna(test["Embarked"].mode()[0], inplace=True)

test["Fare"].fillna(test["Fare"].median(), inplace=True)
#extract input data
train_input_data=train[["Pclass", "Sex", "Age", "Fare", "Embarked"]].values

#convert train_label shape from vector to matrix
train_label=train_label.values.reshape(-1, 1)
#before fit data to learning model, check data shape
print("input_data's shape is: {0}\noutput_data's shape is: {1}".format(train_input_data.shape, train_label.shape))
#make model
lr=LogisticRegression()
lr.fit(train_input_data, train_label)
#predict test data
test_input_data=test[["Pclass", "Sex", "Age", "Fare", "Embarked"]].values
y_pred=lr.predict(test_input_data)
y_pred_pd=pd.Series(y_pred)
submission=pd.concat([test["PassengerId"], y_pred_pd], axis=1)
submission=submission.rename(columns={0: "Survived"})
submission.to_csv("submission_file.csv", index=False)
