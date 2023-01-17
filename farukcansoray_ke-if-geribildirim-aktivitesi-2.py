import time

import os



import pandas as pd # for dataframe operations. 

import numpy as np #for linear algebra operations.

import seaborn as sns # data visualization library

import matplotlib.pyplot as plt # for plotting



from sklearn.model_selection import cross_val_score

from sklearn.externals import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, f1_score, explained_variance_score

from sklearn.metrics import r2_score, mean_squared_log_error



from scipy.stats import kendalltau # we will compare predicted and real results
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



test_id = test['Id']
train.shape
list(train)
ntrain = train.shape[0]

ntest = test.shape[0]
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Özellikler', fontsize=15)

plt.ylabel('Kayıp Verilerin Yüzdesi', fontsize=15)
train = train.drop(columns = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"])

test = test.drop(columns = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"])
numerical_features = train.select_dtypes(exclude = ["object"]).columns

numerical_features_test = test.select_dtypes(exclude = ["object"]).columns
numerical_features_test
numerical_features = train.select_dtypes(exclude = ["object"]).columns

numerical_features_test = test.select_dtypes(exclude = ["object"]).columns

train_num = train[numerical_features]

test_num = test[numerical_features_test]
print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))

train_num = train_num.fillna(train_num.median())

print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))



print("NAs for numerical features in test : " + str(test.isnull().values.sum()))

test_num = test_num.fillna(test_num.median())

print("Remaining NAs for numerical features in test : " + str(test_num.isnull().values.sum()))
train = train_num

train.shape

label = train['SalePrice']



test = test_num
train.SalePrice = np.log1p(train.SalePrice )

y = train.SalePrice
corrmat = train.corr()

f, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(corrmat, vmax=.8, annot=True);
corrmat = train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.6]







plt.figure(figsize=(10,10))

g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
train=train.filter(['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF'],axis=1)

test=test.filter(['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(train,label,test_size = 0.25,random_state= 0)
train.shape
clf = LinearRegression(normalize=True)

scores = cross_val_score(clf, X_train, y_train, cv=5).mean()

scores
clf.fit(X_train, y_train)
pred = clf.predict(test)

pred2 = clf.predict(X_test)
pred2.shape
submission = pd.DataFrame({'Id':test_id,'SalePrice':pred})
submission
filename = 'Price Prediction.csv'

submission.to_csv(filename,index=False)



print('Saved file: ' + filename)
import math

rtrsm = float(format(clf.score(X_train, y_train),'.3f'))

rtesm = float(format(clf.score(X_test, y_test),'.3f'))



print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))

print('Intercept: {}'.format(clf.intercept_))

print('Coefficient: {}'.format(clf.coef_))

clf.score(X_test,y_test)
mean_squared_log_error(y_test, pred2)
explained_variance_score(y_test, pred2)
r2_score(y_test, pred2)