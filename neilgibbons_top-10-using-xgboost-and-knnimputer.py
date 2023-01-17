#import the necessary libraries

import numpy as np

import pandas as pd 

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#import the train and test data

train_df = pd.read_csv("../input/home-data-for-ml-course/train.csv")

test_df = pd.read_csv("../input/home-data-for-ml-course/test.csv")
'''Most of the preprocessing I do here is to fill in missing values, the visualisation below

gives a good overview'''

sns.heatmap(train_df.isnull(), cbar=False)
#find columns in train where more than 40% of the values are NaN

to_drop = [col for col in train_df.columns if (train_df[col].isnull().sum() * 100 / len(train_df[col]))>40]

print("Dropping the following columns: ", to_drop)

#drop these columns from both train and test

train_df.drop(to_drop, axis = 1, inplace=True)

test_df.drop(to_drop, axis = 1, inplace=True)
#concat train and test to save having to repeat preprocessing steps

full_df = pd.concat([train_df,test_df],keys=[0,1])
#Label-encode the categorical columns so that KNNImputer can be used (KNN Imputer requires numeric input)

#first need to put categorical columns in a format that LabelEncoder is happy with

categs = [col for col in full_df.columns if full_df[col].dtype == 'O']

for val in categs:

    full_df[val] = full_df[val].astype(str)

#now these columns are ready to be encoded

from sklearn.preprocessing import LabelEncoder

le_df = full_df.copy(deep=True)

le_df[categs] = le_df[categs].apply(LabelEncoder().fit_transform)
'''Another step before applying KNNImputer: apply MinMaxScaler to numeric columns

This is important to prevent KNNImputer from overlooking valuable information from 

columns with smaller values'''

numeric_cols = [val for val in le_df.columns if val not in categs]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

for col in numeric_cols:

    le_df[numeric_cols]=scaler.fit_transform(le_df[numeric_cols])
#finally, dropping SalePrice and Id since KNNImputer doesn't need these

le_df.drop('SalePrice',axis=1,inplace=True)

le_df.drop('Id',axis=1,inplace=True)
#check everything is correct before applying the Imputer

le_df.head()
#finally time to use KNNImputer to create a new dataframe 

from sklearn.impute import KNNImputer

knn = KNNImputer(n_neighbors=15)

knn.fit(le_df[le_df.columns])

imputed_df = pd.DataFrame(knn.transform(le_df[le_df.columns]),columns=le_df.columns)
'''A satisfying visualisation to show all missing values (represented by white lines earlier) 

have been filled'''

sns.heatmap(imputed_df.isnull(), cbar=False)
#Now to get imputed_df into a format ready for XGBoost

#Split the data back into test and train, adding suffix 'imp' to show they've had missing values imputed

imp_df_train = imputed_df[imputed_df.index<=1459]

imp_df_test = imputed_df[imputed_df.index>1459]
#Adding back in the 'Id' and 'SalePrice' columns I removed earlier

def make_Id(num):

    return num+1

imp_df_train['SalePrice']=train_df['SalePrice'].copy(deep=True)

imp_df_train['Id']=train_df['Id'].copy(deep=True)

imp_df_test['Id']=imp_df_test.index

imp_df_test['Id']=imp_df_test['Id'].apply(make_Id)
#Now ready to use XGBoost to create predictions

import xgboost as xgb

regressor = xgb.XGBRegressor(

    n_estimators=100,

    reg_lambda=1,

    gamma=0,

    max_depth=3

)

regressor.fit(imp_df_train.drop('SalePrice',axis=1),imp_df_train['SalePrice'])

preds =regressor.predict(imp_df_test)
#Lastly, get predictions into a format ready for submission and output the csv

submission = pd.DataFrame(data=(preds), index=test_df.index,columns=['SalePrice'])

submission = submission.join(test_df['Id'])

submission.to_csv('submission.csv', index=False)