import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
train= pd.read_csv("../input/women-health-care-requirements/train Data.csv")
train.head()


def impute_int(df,column):

    df[column] = df[column].fillna(df[column].mean()) 


def impute_categ(df, column):

    df[column] = df[column].fillna(df[column].mode().loc[0]) 
def impute_ordinal(df, column):

    df[column] = df[column].fillna(df[column].mode().loc[0])
# view max of 50 record at a time

pd.set_option('display.max_rows', 50)
def missing_Colums_Percenatage(df):

    missing_values = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending=False) != 0]

    percentage = round((df.isnull().sum().sort_values(ascending = False)*100)/len(df),2)[round((df.isnull().sum().sort_values(ascending = False)*100)/len(df),2) != 0]

    missing_values_df = pd.DataFrame(missing_values)

    percentage_df = pd.DataFrame(percentage)

    missing_values_df.reset_index(level=0, inplace=True)

    percentage_df.reset_index(level=0, inplace=True)

    

    return pd.merge(left=missing_values_df,right= percentage_df, left_on='index', right_on='index')

    

    

missing_Colums_Percenatage(train)
train = train.dropna(axis=1,thresh=13000)
impute_int(train,[col for col in train if col.startswith('n_')])
train[[col for col in train if col.startswith('c_')]]
impute_categ(train,[col for col in train if col.startswith('c_')])
impute_ordinal(train,[col for col in train if col.startswith('o_')])
train[[col for col in train if col.startswith('c_')]].astype(str)

train[[col for col in train if col.startswith('o_')]].astype(str)
train.shape
traindf = pd.get_dummies(train)
traindf.head()
trainlabeldf= pd.read_csv("../input/women-health-care-requirements/train labels.csv")
trainlabeldf['service_a']
completedf = pd.merge(left=traindf,right=pd.DataFrame(trainlabeldf), left_on='id', right_on='id')
completedf.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split
[col for col in completedf if col.startswith('service')]
X = completedf.drop([col for col in completedf if col.startswith('service')], axis=1)

X = X.drop(['id'], axis =1 )

y = completedf[[col for col in completedf if col.startswith('service')]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
regressor = RandomForestClassifier(n_estimators = 100, random_state = 0) 

  

# fit the regressor with x and y data 

clf = regressor.fit(X, y) 
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics

from sklearn.metrics import classification_report

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



#metrics.multilabel_confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))