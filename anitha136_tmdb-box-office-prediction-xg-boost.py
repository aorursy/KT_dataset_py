import numpy as np                                                 

import pandas as pd                                                                                   

import seaborn as sns                                              

%matplotlib inline

sns.set()



from subprocess import check_output

from sklearn import metrics

import xgboost as XGB





import os

print(os.listdir("../input"))



# read data from file

train = pd.read_csv("../input/train.csv") 

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
print("train.shape ",train.shape)

print("test.shape ",test.shape)
train.columns
test.columns
train.describe()
train.isnull().sum()
test.isnull().sum()
test_id=test["id"]

test_id.shape
train.drop(["id","homepage","imdb_id","belongs_to_collection","tagline","original_title","overview","genres","poster_path","production_countries", "production_companies","release_date","spoken_languages","title",'Keywords', 'cast', 'crew'], axis = 1,inplace = True)

test.drop(["id","homepage","imdb_id","belongs_to_collection","tagline","original_title","overview","genres","poster_path","production_countries", "production_companies","release_date","spoken_languages","title",'Keywords', 'cast', 'crew'], axis = 1,inplace = True)
train.columns
test.columns
train.original_language.unique()
from sklearn.preprocessing import LabelEncoder

#label encode region



le = LabelEncoder()

train['original_language'] = le.fit_transform(train['original_language'])

train.head(1)
test['original_language'] = le.fit_transform(test['original_language'])

test.head(1)
train.original_language.unique()
train.status.unique()
train['status'] = le.fit_transform(train['status'])

train.head(1)
test.status.unique()
test.status.mode()
test['status'].fillna("Released", inplace=True)
test.isnull().sum()
test['status'] = le.fit_transform(test['status'])

test.head(1)
test.isnull().any()
np.isnan(train).any()
train.runtime.mode()
train['runtime'].fillna(90, inplace=True)
test.runtime.mode()
test['runtime'].fillna(100, inplace=True)
train.isnull().sum()
test.isnull().sum()
sns.heatmap( train.corr(), annot=True )
sns.distplot(train["budget"],axlabel="Distribution of budget")
sns.distplot(train["revenue"],axlabel="Distribution of revenue")
train.head(1)
X_train=train[[ 'budget', 'original_language', 'popularity', 'runtime', 'status']] #feature columns

y_train=train.revenue #predictor variable
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)

X_train_std = scaler.transform(X_train)

X_train = pd.DataFrame(X_train_std)

X_train.columns = [ 'budget', 'original_language', 'popularity', 'runtime', 'status']

X_train.head()
y_train.head()
scaler = StandardScaler().fit(test)

test_std = scaler.transform(test)

test = pd.DataFrame(test_std)

test.columns = [ 'budget', 'original_language', 'popularity', 'runtime', 'status']

test.head()
print(X_train.shape)

print(y_train.shape)
xg_reg = XGB.XGBRegressor()

xg_reg.fit(X_train,y_train)

y_pred_train = xg_reg.predict(X_train)
print("Model Evaluation for Random Forest Regressor ")

RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))



print('RMSE for training set is {}'.format(RMSE_train))



yhat = xg_reg.predict(X_train)

SS_Residual = sum((y_train-yhat)**2)

SS_Total = sum((y_train-np.mean(y_train))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)

print("r_squared for train data ",r_squared, " and adjusted_r_squared for train data",adjusted_r_squared)

test.head()
y_pred_test = xg_reg.predict(test)
submission = pd.DataFrame({'id': test_id , 'revenue':y_pred_test})

submission = submission[['id', 'revenue']]

submission.head()
submission.to_csv("submission.csv", index=False)