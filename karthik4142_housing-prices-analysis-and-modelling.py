# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

from scipy.stats import norm

from scipy import stats



import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

%matplotlib inline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')

print(train_df.shape)

print(test_df.shape)
train_df.head()
train_df.describe()
missing_data=pd.DataFrame(train_df.isnull().sum().reset_index())



missing_data.columns=["index","missingcount"]

missing_data=missing_data[missing_data["missingcount"]>0]

missing_data['missingper']=(missing_data['missingcount']/train_df.shape[0])*100

missing_data
train_df.drop(['MiscFeature','PoolQC','Fence','FireplaceQu','Alley','LotFrontage'],axis=1,inplace=True)

print(train_df.shape)
cor=train_df.corr()

cor['SalePrice'].sort_values(ascending=False)[0:20]

list_cor=list(cor['SalePrice'].sort_values(ascending=False)[0:20].index)

final_df=train_df[list_cor]

final_df.shape
#list_cor=list(cor['SalePrice'].sort_values(ascending=False)[0:20].index)

final_df[list_cor].isnull().sum()
print(final_df['GarageYrBlt'].dtype)

print(final_df['MasVnrArea'].dtype)
final_df['GarageYrBlt']=final_df['GarageYrBlt'].fillna(final_df['GarageYrBlt'].mode()[0])

final_df['MasVnrArea']=final_df['MasVnrArea'].fillna(final_df['MasVnrArea'].mode()[0])

final_df[list_cor].isnull().sum()
plt.figure(figsize=(12,8))

sns.distplot(final_df['SalePrice'], color='r')

plt.title('Distribution of Sales Price', fontsize=18)



plt.show()
print(final_df['SalePrice'].skew())

print(final_df['SalePrice'].kurt())
final_df['SalePrice']=np.log(final_df.loc[:,'SalePrice'])

print(final_df['SalePrice'].skew())
plt.figure(figsize=(12,8))

sns.distplot(final_df['SalePrice'], color='r')

plt.title('Distribution of Sales Price', fontsize=18)

plt.show()



fig = plt.figure(figsize=(12,8))

res = stats.probplot(final_df['SalePrice'], plot=plt)

plt.show()
corrmat = final_df.corr()

f, ax = plt.subplots(figsize=(22, 9))

sns.heatmap(corrmat, vmax=.8, square=True,annot=True,cmap='YlOrRd',linewidths=0.2,annot_kws={'size':10})

plt.title("Heat map",fontsize=20)

final_df=final_df.drop(["GarageArea","TotRmsAbvGrd","2ndFlrSF","1stFlrSF","GarageYrBlt"],axis=1)

final_df.shape
corrmat = final_df.corr()

f, ax = plt.subplots(figsize=(22, 9))

sns.heatmap(corrmat, vmax=.8, square=True,annot=True,cmap='YlOrRd',linewidths=0.2,annot_kws={'size':10})

plt.title("Heat map",fontsize=20)

plt.figure(figsize=(10,8))

plt.scatter(x='OverallQual',y='SalePrice',data =final_df)

plt.ylabel("SalesPrice")

plt.xlabel("OverallQual")

plt.title("OverallQual vs SalePice")
plt.figure(figsize=(10,8))

plt.scatter(x='GrLivArea',y='SalePrice',data =final_df)

plt.ylabel("SalesPrice")

plt.xlabel("GrLivArea")

plt.title("GrLivArea vs SalePice")
plt.figure(figsize=(10,8))

plt.scatter(x='GarageCars',y='SalePrice',data =final_df)

plt.ylabel("SalesPrice")

plt.xlabel("GarageCars")

plt.title("GarageCars vs SalePice")
plt.figure(figsize=(10,8))

plt.scatter(x='TotalBsmtSF',y='SalePrice',data =final_df)

plt.ylabel("SalesPrice")

plt.xlabel("TotalBsmtSF")

plt.title("TotalBsmtSF vs SalePice")
finaltest_df=final_df["SalePrice"]

finaltrain_df=final_df.drop("SalePrice",axis=1)

print(finaltrain_df.shape)

print(finaltest_df.shape)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

x_train,x_test,y_train,y_test=train_test_split(finaltrain_df,finaltest_df,test_size=0.3)



lr=LinearRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
score = r2_score(y_test,y_pred)

score
from sklearn.preprocessing import Imputer



my_imputer = Imputer()

train_X = my_imputer.fit_transform(x_train)

test_X = my_imputer.transform(x_test)
from xgboost import XGBRegressor



my_model = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(x_train, y_train, verbose=False)
predictions = my_model.predict(x_test)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))
my_model = XGBRegressor(n_estimators=1000)

my_model.fit(x_train, y_train, early_stopping_rounds=5, 

             eval_set=[(x_test, y_test)], verbose=False)
score = r2_score(y_test,predictions)

score