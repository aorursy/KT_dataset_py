# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

plt.style.use('seaborn')

from scipy.stats import norm, skew

import seaborn as sns
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()

#Tahmin işlemi için gerekli olmadığından, 'Id' sütununu kaldırılır.

test_id = test['Id']

train.drop("Id", axis = 1, inplace = True)

test.drop("Id",axis = 1, inplace = True)
print("Train set size:", train.shape)

print("Test set size:", test.shape)

train['SalePrice'].describe()
plt.figure(figsize=(10, 10))

sns.distplot(train['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.5});
corrmat = train.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=.8, annot=True);
most_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(train[most_corr_features].corr(),annot=True,cmap="Blues")

print(most_corr_features)
# Yukarıdaki grafiği yorumladığımızda, SalePrice(Evin Satış Fiyatı) ile en ilişkili özelliğin OverallQual olduğu görülmektedir.

#Belirlenen eşik korelasyon değerini aşan özellikler, 

correlated_cols = ['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars']

#Yukarıdaki Çıktılara göre outlier(ayrık) veriler tespit edilir ve bu veriler kaldırılır.

# GrlivArea >5000 

# 1stFlrSF > 4000

# TotalBsmtSF >6000

#Belirlenen bu özelliklere göre, bu alanlarda yer alan ayrık veriler tespit edilir.

data = pd.concat([train['SalePrice'], train[correlated_cols]], axis=1)

for lst in correlated_cols:

    sns.pairplot(data, y_vars=['SalePrice'], x_vars=lst)

outliers = {"TotalBsmtSF": 6000,"1stFlrSF": 4000, "GrLivArea": 5000}

#Olusan grafiğe göre outlier(ayrık) veriler temizlenir.

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train = train.drop(train[(train['1stFlrSF']>4000) & (train['SalePrice']<200000)].index)

train = train.drop(train[(train['TotalBsmtSF']>6000) & (train['SalePrice']<200000)].index)





test = test.drop(test[(test['GrLivArea']>4000)].index)

test = test.drop(test[(test['1stFlrSF']>4000)].index)

test = test.drop(test[(test['TotalBsmtSF']>6000)].index)







#data = pd.concat([train['SalePrice'], train[correlated_cols]], axis=1)

#for lst in correlated_cols:

    #sns.pairplot(data, y_vars=['SalePrice'], x_vars=lst)
categorical_features = train.select_dtypes(include=['object']).columns

categorical_features

numerical_features = train.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

train_num = train[numerical_features]

train_cat = train[categorical_features]
total = train_num.isnull().sum().sort_values(ascending=False)# train verisetinde yer alan eksik veri sayısını gösterir.

missing_data = pd.concat([total], axis=1, keys=['Total'])

missing_data.head() # hangi özellikte kaç veri eksik olduğunu gösterir
train_num = train_num.fillna(train_num.median())

total = train_num.isnull().sum().sort_values(ascending=False)# train verisetinde yer alan eksik veri sayısını gösterir.

missing_data = pd.concat([total], axis=1, keys=['Total'])

missing_data.head() # hangi özellikte kaç veri eksik olduğunu gösterir
train.SalePrice = np.log1p(train.SalePrice )

y = train.SalePrice
from scipy.stats import skew 

skewness = train_num.apply(lambda x: skew(x))

skewness.sort_values(ascending=False)
train_cat = pd.get_dummies(train_cat)

train_cat.shape

train_cat.head()

str(train_cat.isnull().values.sum())
train = pd.concat([train_cat,train_num],axis=1)

print(train.shape)
from sklearn.model_selection import train_test_split, cross_val_score

X_train,X_test,y_train,y_test = train_test_split(train,y,test_size = 0.3,random_state= 0)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

train= scaler.fit_transform(train)

n_col=train.shape[1]
from sklearn.decomposition import PCA

pca=PCA(n_components=n_col)

train_components=pca.fit(train)

test_components=pca.fit(X_test)

pca.components_
pca.get_covariance()
from sklearn import preprocessing

from sklearn import utils

from sklearn.ensemble import RandomForestRegressor



lab_enc = preprocessing.LabelEncoder() # bunu açıkla

Y = lab_enc.fit_transform(y)

print(Y.shape)

rf = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)

rf.fit(train,Y)

#print(rf.feature_importances_)
from sklearn.svm import SVR

from sklearn.metrics import r2_score

regressor=SVR(kernel='rbf',gamma='auto')



regressor.fit(X_train,y_train)

test_pre = regressor.predict(X_test)

train_pre = regressor.predict(X_train)



plt.scatter(train_pre, train_pre - y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre,test_pre - y_test, c = "black",  label = "Validation data")

plt.title("SVR regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
from sklearn.metrics import make_scorer

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

lr = LinearRegression()

lr.fit(X_train,y_train)

test_pre = lr.predict(X_test)

train_pre = lr.predict(X_train)



print(r2_score(y_test,test_pre))



plt.scatter(train_pre, train_pre - y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre,test_pre - y_test, c = "black",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
predictx=lr.predict(test)

print(predictx.shape)

my_submission = pd.DataFrame({'Id': test_id, 'SalePrice': predictx})

my_submission.to_csv('submission.csv', index=False)