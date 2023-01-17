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
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train.head()
df_train.describe()
df_train.shape
df_test.shape
test_ID = df_test['Id']
del df_train['Id']

del df_test['Id']
fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)





fig, ax = plt.subplots()

ax.scatter(df_train['GrLivArea'], df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice');
df_train = df_train.drop(df_train[(df_train['TotalBsmtSF']>2800) & (df_train['SalePrice']<600000)].index)





fig, ax = plt.subplots()

ax.scatter(df_train['TotalBsmtSF'], df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = df_train['GarageArea'], y = df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GarageArea', fontsize=13)

plt.show()
df_train = df_train.drop(df_train[(df_train['GarageArea']>1200) & (df_train['SalePrice']<13.0)].index)





fig, ax = plt.subplots()

ax.scatter(df_train['TotalBsmtSF'], df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)

plt.show()
var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
print ("Skew is:", df_train.SalePrice.skew())

plt.hist(df_train.SalePrice, color='blue')

plt.show()
df_train['SalePrice'] = np.log(df_train.SalePrice)

print ("Skew is:", df_train['SalePrice'].skew())

plt.hist(df_train['SalePrice'], color='blue')

plt.show()
sns.distplot(df_train['SalePrice'] , fit=norm);





(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')





fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
corrmat = all_data.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
corrmat = all_data.corr()

least_corr_features = corrmat.index[abs(corrmat["SalePrice"])<0.30]

plt.figure(figsize=(10,10))

g = sns.heatmap(all_data[least_corr_features].corr(),annot=True,cmap="RdYlGn")
least_corr_features
all_data.drop(columns={'3SsnPorch','BedroomAbvGr','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','EnclosedPorch','KitchenAbvGr','LotArea','LowQualFinSF','MSSubClass','MiscVal','MoSold','OverallCond','PoolArea','ScreenPorch','YrSold'},inplace=True)
corrmat = all_data.corr()

most_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(all_data[most_corr_features].corr(),annot=True,cmap="RdYlGn")
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
all_data.drop(columns={'PoolQC','MiscFeature','Alley','Fence','FireplaceQu'},inplace=True)
corr = all_data.corr()



print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')

print (corr['SalePrice'].sort_values(ascending=False)[-5:])
ntrain = df_train.shape[0]

ntest = df_test.shape[0]

y_train = df_train.SalePrice.values

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
all_data['GarageYrBlt']=all_data['GarageYrBlt'].fillna(0)
all_data['MasVnrType']=all_data['MasVnrType'].fillna('None')

all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)
for col in('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):

    all_data[col]=all_data.fillna('None')


all_data['Electrical']=all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
for col in ('BsmtFinSF1','TotalBsmtSF'):

    all_data[col] = all_data[col].fillna(0)
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
for col in ('GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
null_columns=all_data.columns[all_data.isnull().any()]

all_data[null_columns].isnull().sum()
from sklearn.preprocessing import LabelEncoder

cols = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC',  'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir')



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



      

print('Shape all_data: {}'.format(all_data.shape))
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
from scipy import stats

from scipy.stats import norm, skew 

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)
all_data = pd.get_dummies(all_data)

print(all_data.shape)
train = all_data[:ntrain]

test = all_data[ntrain:]
train.head()
test.head()
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
y = df_train['SalePrice']

y_train = y

X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(

                                     train, y_train,

                                     test_size=0.25,

                                     random_state=42

                                     )
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)

import time

from sklearn import metrics



def regression(regr,X_test_sparse,y_test_sparse):

    start = time.time()

    regr.fit(X_train_sparse,y_train_sparse)

    end = time.time()

    rf_model_time=(end-start)/60.0

    print("Time taken to model: ", rf_model_time , " minutes" ) 

    

def regressionPlot(regr,X_test_sparse,y_test_sparse,title):

    predictions=regr.predict(X_test_sparse)

    plt.figure(figsize=(10,6))

    plt.scatter(predictions,y_test_sparse,cmap='plasma')

    plt.title(title)

    plt.show()

    

    print('RMSE:', np.sqrt(metrics.mean_squared_error(np.log1p(y_test_sparse), np.log1p(predictions))))

    
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)



score = rmsle_cv(GBoost)



regression(GBoost,X_test_sparse,y_test_sparse)

regressionPlot(GBoost,X_test_sparse,y_test_sparse,"Gradient Boosting Regression")
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)



score = rmsle_cv(model_xgb)



regression(model_xgb,X_test_sparse,y_test_sparse)

regressionPlot(model_xgb,X_test_sparse,y_test_sparse,"XGBoost")
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
print(' RMSLE score:')

print(rmsle(y_train,xgb_train_pred))
pred = xgb_pred
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = pred

sub.to_csv('submission.csv',index=False)