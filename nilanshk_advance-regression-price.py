# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy.stats as stats

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

target=['SalePrice']

X_train_data=train_data.drop(['SalePrice'],axis=1)

test_data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
testID=test_data["Id"]
data=pd.concat([X_train_data, test_data], axis=0)
nan_cols = [i for i in data.columns if data[i].isnull().any()]

data[nan_cols].isnull().sum()
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] =16.0

fig_size[1] = 10.0

data[nan_cols].isnull().sum().plot(kind='bar')

plt.plot(data[nan_cols].isnull().sum())

plt.show()
data=data.drop(['Id'],axis=1)
null_numerical_cols = [nan_cols for nan_cols in data[nan_cols] if 

                data[nan_cols].dtype in ['int64', 'float64']]
categorical_cols_null = [nan_cols for nan_cols in data[nan_cols] if

                    data[nan_cols].dtype == "object"]
columns=data.columns
numerical_cols = [columns for columns in data[columns] if 

                data[columns].dtype in ['int64', 'float64']]
categorical_cols = [columns for columns in data[columns] if

                    data[columns].nunique() < 26 and 

                    data[columns].dtype == "object"]
var_with_meaning=['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'FireplaceQu', 'Fence', 'MiscFeature', 'PoolQC']

data['BsmtQual'].fillna("No Basement",inplace=True)

data['BsmtCond'].fillna("No Basement",inplace=True)

data['BsmtExposure'].fillna("No Basement",inplace=True)

data['BsmtFinType1'].fillna("No Basement",inplace=True)

data['BsmtFinType2'].fillna("No Basement",inplace=True)

data['GarageCond'].fillna("No Garage",inplace=True)

data['GarageQual'].fillna("No Garage",inplace=True)

data['GarageFinish'].fillna("No Garage",inplace=True)

data['GarageType'].fillna("No Garage",inplace=True)

data['FireplaceQu'].fillna("No Fireplace",inplace=True)

data['Fence'].fillna("No Fence",inplace=True)

data['MiscFeature'].fillna("No Misc",inplace=True)

data['PoolQC'].fillna("No Pool",inplace=True)

data['Alley'].fillna("No Alley",inplace=True)



nan_cols = [i for i in data.columns if data[i].isnull().any()]

data[nan_cols].isnull().sum()
Y_data=train_data['SalePrice']

X_data=data
Y_data=pd.DataFrame(Y_data)

Y_data
columns_new=X_data.columns
numerical_cols_new=[columns_new for columns_new in X_data[columns_new] if 

                X_data[columns_new].dtype in ['int64', 'float64']]
from sklearn.impute import SimpleImputer

numerical_transformer = SimpleImputer(strategy='mean')

X_data_new=X_data[numerical_cols_new]

X_data[numerical_cols_new]=numerical_transformer.fit_transform(X_data[numerical_cols_new])
X_data['MasVnrType'].fillna('None',inplace=True)

X_data['Electrical'].fillna('SBrkr',inplace=True)
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy='most_frequent')

X_data[nan_cols]=imputer.fit_transform(X_data[nan_cols])
categorical_cols = [cname for cname in X_data.columns if

                    X_data[cname].nunique() < 26 and 

                    X_data[cname].dtype == "object"]
from sklearn.preprocessing import LabelEncoder

categorical_cols = [cname for cname in X_data.columns if

                    X_data[cname].nunique() < 26 and 

                    X_data[cname].dtype == "object"]

X_data_encoded = X_data[categorical_cols]

label_encoder=LabelEncoder()

for i, col in enumerate(X_data[categorical_cols].columns):

    X_data_encoded[col]=label_encoder.fit_transform(X_data_encoded[col])

######inverse_transform()  for reverseing the transformation  ########
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

X_data_encoded_ordinal=X_data

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(X_data[c].values)) 

    X_data_encoded_ordinal[c]= lbl.transform(list(X_data[c].values))



# shape        

print('Shape all_data: {}'.format(X_data.shape))
numerical_cols=[columns_new for columns_new in X_data[columns_new] if 

                X_data[columns_new].dtype in ['int64', 'float64']]



X_data_num=X_data[numerical_cols]
X_preprocess_data3=X_data_encoded_ordinal
from scipy.stats import norm, skew #for some statistics

# Check the skew of all numerical features

skewed_feats = X_preprocess_data3[numerical_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness.plot()
#Transformed Data



skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    X_preprocess_data3[feat] = boxcox1p(X_preprocess_data3[feat], lam)
X_dummy=pd.concat([X_preprocess_data3.iloc[0:1460,:][numerical_cols],Y_data],axis=1)

correlation=X_dummy.corr()['SalePrice']

corr_colums=correlation.sort_values(ascending=False)[:20]

corr_colums=pd.DataFrame(corr_colums)

corr_colums
from sklearn import svm

from sklearn.datasets import make_classification

from sklearn.feature_selection import SelectKBest, f_regression, f_classif

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split



XX_data=X_preprocess_data3.iloc[0:1460,:]

label_encoder=LabelEncoder()

for i, col in enumerate(X_preprocess_data3.iloc[0:1460,:].columns):

    XX_data[col]=label_encoder.fit_transform(X_preprocess_data3.iloc[0:1460,:][col])

    

sel2=f_classif(XX_data,Y_data)

p_values2=pd.Series(sel2[1])

p_values2.index=XX_data.columns

p_values2.sort_values(ascending=True, inplace=True)
sel=f_regression(XX_data,Y_data)

p_values=pd.Series(sel[1])

p_values.index=XX_data.columns

p_values.sort_values(ascending=True, inplace=True)
p_values2.plot.bar(figsize=(16,5))
p_values.plot.bar(figsize=(16,5))
p_values2=p_values2[p_values2<0.05]
p_values=p_values[p_values<0.05]
X_preprocess_data5=X_preprocess_data3[p_values2.index]
X_preprocess_data4=X_preprocess_data3[p_values.index]
variables=['']
X_preprocess_data32= pd.get_dummies(X_preprocess_data3)

print(X_preprocess_data32.shape)
X_Train=X_preprocess_data32.iloc[0:1460,:]
X_test=X_preprocess_data32.iloc[1460:2919,:]
plt.subplots(figsize=(12,9))

sns.distplot(Y_data['SalePrice'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(Y_data['SalePrice'])



# plot with the distribution



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(Y_data['SalePrice'], plot=plt)

plt.show()
#we use log function which is in numpy

Y_data['SalePrice'] = np.log1p(Y_data['SalePrice'])



#Check again for more normal distribution



plt.subplots(figsize=(12,9))

sns.distplot(Y_data['SalePrice'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(Y_data['SalePrice'])



# plot with the distribution



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot

fig = plt.figure()

stats.probplot(Y_data['SalePrice'], plot=plt)

plt.show()
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

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from vecstack import stacking
models = [

    xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =0, nthread = -1),

    RandomForestRegressor(n_estimators=1000),

    GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5),

    lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11),

    make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

    

]
S_train, S_test = stacking(models,                   

                           X_Train, Y_data, X_test,   

                           regression=True, 

                           mode='oof_pred_bag', 

       

                           needs_proba=False,

         

                           save_dir=None, 

            

                           metric=mean_squared_error, 

    

                           n_folds=4, 

                 

                           stratified=True,

            

                           shuffle=True,  

            

                           random_state=0,    

         

                           verbose=2)
model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =0, nthread = -1)

    

model = model.fit(S_train, Y_data)

y_pred = np.expm1(model.predict(S_test))

sub = pd.DataFrame()

sub['Id'] = testID

sub['SalePrice'] = y_pred

sub.to_csv('submission.csv',index=False)