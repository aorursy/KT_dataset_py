# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_dataset = '../input/train_V2.csv'
test_dataset = '../input/test_V2.csv'
train_df = pd.read_csv(train_dataset)
train_df.head()
train_df.isnull().sum() #Check if there are any null
train_df = train_df.dropna() #remove nulls from datasets
target = train_df['winPlacePerc'] #target variable to find
features = train_df.drop(['winPlacePerc'],axis=1) #input features
features.head()
refine_features = features.drop(['Id','groupId','matchId'],axis=1) #drop unnecessary features
refine_features.info()
train_df.corr()['winPlacePerc'].sort_values().plot(kind='bar',figsize=(11,7))
#Convert categorical variables to numerical by encoding of 0 and 1
refine_features = pd.get_dummies(refine_features)
refine_features.info()
#calculate vif of each column(feature)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['vif factor'] = [variance_inflation_factor(refine_features.values,i) for i in range(refine_features.shape[1])]
vif['features'] = refine_features.columns
vif.sort_values(by=['vif factor'],ascending=False)


#dropping columns based on hig VIF
list_of_drop_cols = ['maxPlace','numGroups','matchType_squad-fpp','matchType_duo-fpp','winPoints','matchType_solo-fpp','rankPoints','matchType_squad','matchType_duo','matchType_solo']
refine_features = refine_features.drop(list_of_drop_cols,axis = 1)
refine_features.shape
#Create cross validation test sets to check if model is trained well or not.
from sklearn.model_selection import train_test_split
Xtrain,Xvalidation,Ytrain,Yvalidation = train_test_split(refine_features,target,test_size=0.25)
print(len(Xtrain))
print(len(Ytrain))
print(len(Xvalidation))
print(len(Yvalidation))
#GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import r2_score

linear = GradientBoostingRegressor(learning_rate = 1.0, n_estimators = 100, max_depth = 4)
linear.fit(Xtrain,Ytrain)
pred = linear.predict(Xvalidation)
print(r2_score(Yvalidation,pred))


 
test_df = pd.read_csv(test_dataset)
test_features = test_df.drop(['Id','groupId','matchId'],axis = 1)
test_features = pd.get_dummies(test_features)
test_features = test_features.drop(['maxPlace','numGroups','matchType_squad-fpp'
                              ,'matchType_duo-fpp','winPoints','matchType_solo-fpp','rankPoints',
                              'matchType_squad','matchType_duo','matchType_solo'],axis=1)

Xtest = test_features
test_features.shape
from sklearn.ensemble import GradientBoostingRegressor


linear = GradientBoostingRegressor()
linear.fit(Xtrain,Ytrain)
pred = linear.predict(Xtest)
pred[:10]

pred_df = pd.DataFrame(pred,test_df['Id'],columns=['WinPlacePerc'])
pred_df.head()


pred_df.to_csv('sample_submission.csv')


