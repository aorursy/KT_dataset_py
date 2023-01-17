# importing the librarires



import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error

from math import sqrt

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

import statsmodels.api as sm

from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

import pickle

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.svm import SVR

from sklearn.linear_model import Lasso, Ridge, ElasticNet

# importing the training dataset



train = pd.read_csv('../input/big-mart-sales/bigmart_train.csv')
train.head()
train.shape
# 1. Missing Values in train dataset

# 2. defining a function for missing values



def missing_value(train):

    total = train.isnull().sum().sort_values(ascending=False)

    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])

    return missing_data



missing_value(train)
train['Item_Fat_Content'].unique()

#notice Low fat, Low Fat, LF are all the same variable
train['Outlet_Establishment_Year'].unique()
train['Outlet_Age'] = 2018 - train['Outlet_Establishment_Year']

train.head()
#train['Outlet_Size'].unique()
train.describe()
train['Item_Fat_Content'].value_counts()
train['Outlet_Size'].value_counts()
train['Outlet_Size'].mode()[0]
# fill the na for outlet size with medium

#train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
# fill the na for item weight with the mean of weights

train['Item_Weight'].interpolate(inplace=True)
train['Item_Visibility'].hist(bins=20)
# delete the observations



Q1 = train['Item_Visibility'].quantile(0.25)

Q3 = train['Item_Visibility'].quantile(0.75)

IQR = Q3 - Q1

filt_train = train.query('(@Q1 - 1.5 * @IQR) <= Item_Visibility <= (@Q3 + 1.5 * @IQR)')
filt_train
filt_train.shape, train.shape
train = filt_train

train.shape
#train['Item_Visibility'].value_counts()
#creating a category

train['Item_Visibility_bins'] = pd.cut(train['Item_Visibility'], [0.000, 0.065, 0.13, 0.2], labels=['Low Viz', 'Viz', 'High Viz'])
train['Item_Visibility_bins'].value_counts()
train['Item_Visibility_bins'].unique()
train['Item_Visibility_bins'] = train['Item_Visibility_bins'].replace(np.nan,'Low Viz',regex=True)

train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace('reg', 'Regular')
train.head()
#le = LabelEncoder()
train['Item_Fat_Content'].unique()
train.replace({'Low Fat':0, 'Regular':1}, inplace=True)
#train['Item_Fat_Content'] = le.fit_transform(train['Item_Fat_Content'])
train.replace({'High Viz':2, 'Viz':1,'Low Viz':0}, inplace=True)
train['Outlet_Size'].unique()
train.replace({'High':2, 'Medium':1, 'Small':0}, inplace=True)
train['Outlet_Location_Type'].unique()
train.replace({'Tier 1':2,'Tier 2':1,'Tier 3':0},inplace = True)

print(train["Outlet_Location_Type"].value_counts())
# create dummies for outlet type
item_type = pd.get_dummies(train["Item_Type"], drop_first= True)

outlet_indentifier= pd.get_dummies(train["Outlet_Identifier"], drop_first= True)

#Outlet_Type= pd.get_dummies(train["Outlet_Type"], drop_first= True)

del train["Item_Type"]

del train["Outlet_Identifier"]

#del train["Outlet_Type"]

del train["Item_Identifier"]
train = pd.concat([train,item_type,outlet_indentifier], axis=1)

pd.set_option('display.max_columns',None)

train.head()
outlet_Type= pd.get_dummies(train["Outlet_Type"], drop_first= True)

del train["Outlet_Type"]

train = pd.concat([train,outlet_Type], axis=1)

pd.set_option('display.max_columns',None)

train.head()
train.dtypes


train = train.drop(['Outlet_Establishment_Year','Item_Visibility'], axis=1)
train.columns
from statsmodels.imputation import mice 



train= mice.MICEData(train).data
# build the linear regression model

X = train.drop('Item_Outlet_Sales', axis=1)

y = train.Item_Outlet_Sales
test = pd.read_csv('../input/big-mart-sales/bigmart_test.csv')

#test['Outlet_Size'] = test['Outlet_Size'].fillna('Medium')
test['Item_Visibility_bins'] = pd.cut(test['Item_Visibility'], [0.000, 0.065, 0.13, 0.2], labels=['Low Viz', 'Viz', 'High Viz'])
test['Item_Weight'].interpolate(inplace=True)
test['Item_Visibility_bins'] = test['Item_Visibility_bins'].fillna('Low Viz')

test['Item_Visibility_bins'].head()
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')

test['Item_Fat_Content'] = test['Item_Fat_Content'].replace('reg', 'Regular')
#test['Item_Fat_Content'] = le.fit_transform(test['Item_Fat_Content'])
#test['Item_Visibility_bins'] = le.fit_transform(test['Item_Visibility_bins'])
#test['Outlet_Size'] = le.fit_transform(test['Outlet_Size'])
#test['Outlet_Location_Type'] = le.fit_transform(test['Outlet_Location_Type'])
#test['Outlet_Age'] = 2018 - test['Outlet_Establishment_Year']
'''dummy = pd.get_dummies(test['Outlet_Type'])

test = pd.concat([test, dummy], axis=1)

'''
#X_test = test.drop(['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Type','Outlet_Establishment_Year'], axis=1)
#X.columns, X_test.columns
import seaborn as sns



plt.figure(figsize=(30,30))

sns.heatmap(X.corr(), cmap='RdYlGn', annot = True)
del X["OUT027"]

del X["OUT018"]
from sklearn import model_selection

xtrain,xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=0.3,random_state=0)
sc = StandardScaler()

sc.fit_transform(xtrain)

sc.transform(xtest)
pca = PCA(n_components=0.95)

pca.fit_transform(xtrain)

pca.transform(xtest)
lr = LinearRegression()
lr.fit(xtrain, ytrain)

print(lr.coef_)

lr.intercept_

y_pred = lr.predict(xtest)

print(sqrt(mean_squared_error(ytest, y_pred)))
from sklearn.linear_model import Ridge

ridgeReg = Ridge(alpha=0.001, normalize=True)

ridgeReg.fit(xtrain,ytrain)

print(sqrt(mean_squared_error(ytrain, ridgeReg.predict(xtrain))))

print(sqrt(mean_squared_error(ytest, ridgeReg.predict(xtest))))

print('R2 Value/Coefficient of Determination: {}'.format(ridgeReg.score(xtest, ytest)))

print(pearsonr(ytest, ridgeReg.predict(xtest)))

from scipy.stats import pearsonr

from sklearn.linear_model import Lasso

lassoreg = Lasso(alpha=0.001, normalize=True)

lassoreg.fit(xtrain, ytrain)



print(sqrt(mean_squared_error(ytrain, lassoreg.predict(xtrain))))

print(sqrt(mean_squared_error(ytest, lassoreg.predict(xtest))))

print('R2 Value/Coefficient of Determination: {}'.format(lassoreg.score(xtest, ytest)))

print(pearsonr(ytest, lassoreg.predict(xtest)))

sns.scatterplot(ytest, lassoreg.predict(xtest))
from sklearn.linear_model import ElasticNet

Elas = ElasticNet(alpha=0.001, normalize=True)

Elas.fit(xtrain, ytrain)



print(sqrt(mean_squared_error(ytrain, Elas.predict(xtrain))))

print(sqrt(mean_squared_error(ytest, Elas.predict(xtest))))

print('R2 Value/Coefficient of Determination: {}'.format(Elas.score(xtest, ytest)))

print(pearsonr(ytest, Elas.predict(xtest)))

sns.scatterplot(ytest, Elas.predict(xtest))
from sklearn.ensemble import GradientBoostingRegressor



gbr = GradientBoostingRegressor()

gbr.fit(xtrain,ytrain)

print(sqrt(mean_squared_error(ytrain, gbr.predict(xtrain))))

print(sqrt(mean_squared_error(ytest, gbr.predict(xtest))))

print('R2 Value/Coefficient of Determination: {}'.format(gbr.score(xtest, ytest)))
print(pearsonr(ytest, gbr.predict(xtest)))

sns.scatterplot(ytest, gbr.predict(xtest))
from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(xtrain, ytrain)



print(sqrt(mean_squared_error(ytrain, xgb.predict(xtrain))))

print(sqrt(mean_squared_error(ytest, xgb.predict(xtest))))

print('R2 Value/Coefficient of Determination: {}'.format(xgb.score(xtest, ytest)))
print(pearsonr(ytest, xgb.predict(xtest)))

sns.scatterplot(ytest, xgb.predict(xtest))