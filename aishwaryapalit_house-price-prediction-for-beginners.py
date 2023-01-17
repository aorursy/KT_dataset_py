# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
# Load Data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#top five row of data
train.head()
test.head()
#dimension of train data
train.shape
#dimension of test data
test.shape
train.info()
sns.distplot(train['SalePrice']);
#skewness
print("Skewness: %f" % train['SalePrice'].skew())
#To remove the skewness we use the log function
SalePriceLog = np.log(train['SalePrice'])
SalePriceLog.skew()
#Plot after adjusted skewness
sns.distplot(SalePriceLog);
SalePrice = SalePriceLog
#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(50,20))
sns.heatmap(corrmat, vmax=0.9, square=True, annot=True)
Num=corrmat['SalePrice'].sort_values(ascending=False).head(10).to_frame()

Num
#missing data
total = train.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([total], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#missing data
total = test.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([total], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#visulize missing value using sns plot
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=total.index, y=total)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','MasVnrType'], axis=1 ,inplace=True)
test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','MasVnrType'], axis=1 ,inplace=True)
# missing value treatment for continuous variable
for col in ('LotFrontage','GarageYrBlt','GarageCars','BsmtFinSF1','TotalBsmtSF','GarageArea','BsmtFinSF2','BsmtUnfSF','LotFrontage','GarageYrBlt','BsmtFullBath','BsmtHalfBath'):
    train[col]=train[col].fillna(train[col].mean())
    test[col]=test[col].fillna(test[col].mean())
# missing value treatment for categorical variable
for col in ('BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrArea', 'Electrical','Exterior2nd','Exterior1st','KitchenQual','Functional','SaleType','Utilities','MSZoning','BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrArea', 'Electrical'):
    test[col]=test[col].fillna(test[col].mode()[0])
    train[col]=train[col].fillna(train[col].mode()[0])
# checking if is there any missing variable left
train.isnull().sum().max()
# checking if is there any missing variable left
test.isnull().sum().max()
list_of_numerics=train.select_dtypes(include=['float','int']).columns
types= train.dtypes

outliers= train.apply(lambda x: sum(
                                 (x<(x.quantile(0.25)-1.5*(x.quantile(0.75)-x.quantile(0.25))))|
                                 (x>(x.quantile(0.75)+1.5*(x.quantile(0.75)-x.quantile(0.25))))
                                 if x.name in list_of_numerics else ''))


explo = pd.DataFrame({'Types': types,
                      'Outliers': outliers}).sort_values(by=['Types'],ascending=False)
explo.transpose()
fig, axes = plt.subplots(1,2, figsize=(12,5))

ax1= sns.scatterplot(x='GrLivArea', y='SalePrice', data= train,ax=axes[0])
ax2= sns.boxplot(x='GrLivArea', data= train,ax=axes[1])
#removing outliers recomended by author
train= train[train['GrLivArea']<4000]
#test= test[test['GrLivArea']<4000]
train['MSSubClass'] = train['MSSubClass'].apply(str)
train['YrSold'] = train['YrSold'].astype(str)

test['MSSubClass'] = test['MSSubClass'].apply(str)
test['YrSold'] = test['YrSold'].astype(str)
categorial_features_train = train.select_dtypes(include=[np.object])
categorial_features_train.head(2)
categorial_features_test = test.select_dtypes(include=[np.object])
categorial_features_test.head(2)
##Label Encoding
from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()

label_encoders = {}
for column in categorial_features_train:
    label_encoders[column] = LabelEncoder()
    train[column] = label_encoders[column].fit_transform(train[column]) 
##Label Encoding
from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()

label_encoders = {}
for column in categorial_features_test:
    label_encoders[column] = LabelEncoder()
    test[column] = label_encoders[column].fit_transform(test[column]) 
# dividing into dependent and independent variable data set
xtrain = train.drop('SalePrice', axis = 1)
ytrain = train['SalePrice']
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import xgboost as xgb
model1 = LinearRegression()
model1.fit(xtrain, ytrain)
# score the model
model1.score(xtrain,ytrain)
model2 = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model2.fit(xtrain,ytrain)
model2.score(xtrain,ytrain)
model3 = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model3.fit(xtrain, ytrain)
model3.score(xtrain, ytrain)
pred_1=model1.predict(test)
pred_2=model2.predict(test)
pred_3=model3.predict(test)
final_pred = (pred_1+pred_2+pred_3)/3
final_pred
sample_sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
sample_sub.head()
sample_sub['SalePrice'] = final_pred
sample_sub.to_csv('final_submission1.csv', index=False)