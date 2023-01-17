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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.special import boxcox1p

from scipy.stats import skew

from sklearn.linear_model import Lasso, LassoCV

from sklearn.preprocessing import StandardScaler, RobustScaler
#bring in the six packs

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test_c=pd.read_csv("../input/test.csv")

print(train.shape)

print(test.shape)
#check the decoration

train.columns
#descriptive statistics summary

train['SalePrice'].describe()

houses=pd.concat([train,test], sort=False)

houses.select_dtypes(include='object').head(10)
houses.sample(5)
houses.select_dtypes(include=['float','int']).head(10)
#How many missing values does the dataset have?

print(train.isnull().sum().sum())

print(test.isnull().sum().sum())
#Let's plot these missing values vs column_names

houses.select_dtypes(include='object').isnull().sum()[houses.select_dtypes(include='object').isnull().sum()>0]
numerical = houses.select_dtypes(include=['int','float']).isnull()

numerical_sum = numerical.sum()[houses.select_dtypes(include=['int','float']).isnull().sum()>0]

numerical_sum
# Which columns have the most missing values?

def missing_data(df):

    total = df.isnull().sum()

    percent = (df.isnull().sum()/train.isnull().count()*100)

    missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in df.columns:

        dtype = str(df[col].dtype)

        types.append(dtype)

    missing_values['Types'] = types

    missing_values.sort_values('Total',ascending=False,inplace=True)

    return(np.transpose(missing_values))

missing_data(train)
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

           'PoolQC','Fence','MiscFeature'):

    train[col]=train[col].fillna('None')

    test[col]=test[col].fillna('None')
for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):

    train[col]=train[col].fillna(train[col].mode()[0])

    test[col]=test[col].fillna(train[col].mode()[0])


print(train.isnull().sum().sum())
for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):

    train[col]=train[col].fillna(0)

    test[col]=test[col].fillna(0)
train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())

test['LotFrontage']=test['LotFrontage'].fillna(train['LotFrontage'].mean())
print(test.isnull().sum().sum())
train.hist(bins=50, figsize=(20,15))

plt.tight_layout(pad=0.4)

plt.show()
test.hist(bins=50, figsize=(20,15))

plt.tight_layout(pad=0.4)

plt.show()
plt.figure(figsize=[30,15])

sns.heatmap(train.corr(), annot=True)
train_eda = train.copy()

label_col = 'SalePrice'

base_color = sns.color_palette()[0]

plt.figure(figsize=(20,15))

plt.xticks(rotation=45)

sns.boxplot(data = train_eda, x = 'BedroomAbvGr', y = 'SalePrice', color = base_color);
''' 

train_eda.plot(kind="scatter", x=label_col, y="GarageCars", alpha=0.4,

             s=train_eda["BedroomAbvGr"], label="BedroomAbvGr", figsize=(20,15),

             c="YrSold", cmap=plt.get_cmap("jet"), colorbar=True,)

plt.axis([0, 400000, 0, 1200])

plt.legend();

''' 
len_train=train.shape[0]

print(train.shape)
#from 2 features high correlated, removing the less correlated with SalePrice

train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)

test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
#removing outliers recomended by author

train = train[train['GrLivArea']<4000]
len_train=train.shape[0]

print(train.shape)
houses=pd.concat([train,test], sort=False)
houses['MSSubClass']=houses['MSSubClass'].astype(str)
skew=houses.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skew_df=pd.DataFrame({'Skew':skew})

skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]

skewed_df.head(10)
skewed_df.index
train=houses[:len_train]

test=houses[len_train:]
lam=0.1

for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',

       'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',

       'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',

       'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces',

       'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt',

       'GarageYrBlt'):

    train[col]=boxcox1p(train[col],lam)

    test[col]=boxcox1p(test[col],lam)
train['SalePrice']=np.log(train['SalePrice'])
houses=pd.concat([train,test], sort=False)

houses=pd.get_dummies(houses)
train=houses[:len_train]

test=houses[len_train:]
train.drop('Id', axis=1, inplace=True)

test.drop('Id', axis=1, inplace=True)
x=train.drop('SalePrice', axis=1)

y=train['SalePrice']

test=test.drop('SalePrice', axis=1)

sc=RobustScaler()

x=sc.fit_transform(x)

test=sc.transform(test)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb



n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

#model = lasso

x.shape
lasso.fit(x,y)
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
''' 

lasso.fit(x,y)



y_train_las = lasso.predict(x)

y_test_las = lasso.predict(test)



# Plot residuals

plt.scatter(y_train_las, y_train_las - y, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_las, y_test_las - test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Lasso regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



'''
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

model = ENet

ENet.fit(x,y)
score = rmsle_cv(ENet)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
pred=model.predict(test)

preds=np.exp(pred)
output=pd.DataFrame({'Id':test_c.Id, 'SalePrice':preds})

output.to_csv('submission.csv', index=False)
output.head(15)