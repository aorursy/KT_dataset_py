# import Libraries

import pandas as pd

import numpy as np

import matplotlib as pyplot

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# to see all the comands result in a single kernal 

%load_ext autoreload

%autoreload 2

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# to increase no. of rows and column visibility in outputs

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
#Import data

train = pd.read_csv(r'../input/house-price-prediction-challenge-machine-hack/Train.csv')

test = pd.read_csv(r'../input/house-price-prediction-challenge-machine-hack/Test.csv')

sample = pd.read_csv(r'../input/house-price-prediction-challenge-machine-hack/sample_submission.csv')
# Having a look at data and its shape 

train.head()

test.head()

train.shape

test.shape

sample.shape
#Conerting categorical variable to numeric

train['BHK_OR_RK']=train['BHK_OR_RK'].replace({'BHK':0,'RK':1})

train['POSTED_BY']=train['POSTED_BY'].replace({'Owner':0,'Dealer':1,'Builder':2})

test['BHK_OR_RK']=test['BHK_OR_RK'].replace({'BHK':0,'RK':1})

test['POSTED_BY']=test['POSTED_BY'].replace({'Owner':0,'Dealer':1,'Builder':2})
# to check type of columns and identify whether missing values exist or not

train.info()

test.info()
# found out that there is no. missing value and only one address as object type variable

# Target varibale distribution 

train['TARGET(PRICE_IN_LACS)'].plot(kind = 'density', title = 'Price Distribution')
# Transforming target varible(log transformation), because target is to optimize Root mean square log error

# and checking log transformed varibale distribution

train['TARGET_log']=np.log1p(train['TARGET(PRICE_IN_LACS)'])

train['TARGET_log'].plot(kind = 'density', title = 'log of Price Distribution')
# Check Dublicacy

duplicateRowsDF = train[train.duplicated()]

print("Duplicate Rows except first occurrence based on all columns are :")

duplicateRowsDF.shape
# Removing Dublicacy

train.drop_duplicates(keep = False, inplace = True,ignore_index=True) 
# Analysing distribution in Categorical varibales 

cat_cols = ['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'BHK_OR_RK', 'RESALE']

fig, axes = plt.subplots(1, 6, figsize=(24, 10))



for i, c in enumerate(['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'BHK_OR_RK', 'RESALE']):

    _ = train[c].value_counts()[::-1].plot(kind = 'pie', ax=axes[i], title=c, autopct='%.0f', fontsize=18)

    _ = axes[i].set_ylabel('')

    

_ = plt.tight_layout()
# we found out that 99% of houses are either 1,2,3,or 4 BHK 

# almost all the houses are BHK 

# to get exact values of distribution 

cat_col=['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'BHK_OR_RK', 'RESALE']

for col in cat_col:

 train[col].value_counts()/len(train)
# Analysing distribution in Numeric varibales, sqrt ft with target

sns.scatterplot(x=np.log1p(train['SQUARE_FT']), y=train['TARGET_log'])
# Checking correlation

plt.figure(figsize=(15, 8))

sns.heatmap(train.corr(),annot=True)
#ready ro move and under construction have correlation -1 so both giving same information, we remove one of model
train['sq_per_room']=train['SQUARE_FT']/train['BHK_NO.']

test['sq_per_room']=test['SQUARE_FT']/test['BHK_NO.']
# Extracting name of city and locality of house

import re

def city(address):

 city_name=address.split(',')[-1]

 return city_name

def locality(address):

 locality=address.split(',')[-2]

 return locality

train['loc']=train['ADDRESS'].apply(lambda x : locality(x))

train['City']=train['ADDRESS'].apply(lambda x : city(x))

test['loc']=test['ADDRESS'].apply(lambda x : locality(x))

test['City']=test['ADDRESS'].apply(lambda x : city(x))
# Mean Encoding city (adding a new varibale City_mean which value is equal to average price of house in that city)

Encoding = train.groupby('City')['TARGET(PRICE_IN_LACS)'].mean()

train['City_mean']= train.City.map(Encoding )

test['City_mean']= test.City.map(Encoding )
# Selecting Columns to take in the model

col=['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'BHK_OR_RK', 'SQUARE_FT','LONGITUDE', 'LATITUDE', 'RESALE','City_mean','sq_per_room']
# sub_Train validation set split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(train, test_size=.25, random_state=150303,shuffle=True)
# Model Fitting on sub_train set and evaluating score on validation set 

from lightgbm import LGBMRegressor

lgbcl = LGBMRegressor(n_estimators=1000, importance_type='gain')

lgbcl= lgbcl.fit(X_train[col],X_train['TARGET_log'],categorical_feature=cat_col,eval_set=(X_val[col],X_val['TARGET_log']),verbose=100,early_stopping_rounds=100)

y_predict = lgbcl.predict(X_val[col])

np.sqrt(mean_squared_error(X_val['TARGET_log'],y_predict))
# Checking Feature Importance

feat_importances = pd.Series(lgbcl.feature_importances_, index=col)

feat_importances.nlargest(8).plot(kind='barh')

plt.show()
# Fitting model on complete training set and Predicting on test set

lgbcl= lgbcl.fit(train[col], train['TARGET_log'])

lgb_pred = lgbcl.predict(test[col])

sample['TARGET(PRICE_IN_LACS)']=np.abs((np.exp(lgb_pred)-1))

sample.to_csv('lgbm_.csv',index=False)