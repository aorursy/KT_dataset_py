#Import libraries.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



from sklearn.feature_extraction import DictVectorizer

import statsmodels.regression.linear_model as sm

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

%matplotlib notebook
#Import dataset.

train = pd.read_csv('../input/home-data-for-ml-course/train.csv', delimiter=',')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv', delimiter=',')



train.set_index('Id', inplace=True)

test.set_index('Id', inplace=True)



print('train set size : {}'.format(train.shape))

print('test set size : {}'.format(test.shape))
train.columns[train.dtypes != 'object']
numerical = train.select_dtypes(exclude='object').drop('SalePrice', axis=1).copy()

numerical
numerical.shape
# #Repartition of values for each numerical features.

# for elt in numerical.columns:

#     print("{} -> {}".format(elt, numerical[elt].unique()), end='\n\n')
# Histogram.

fig = plt.figure(figsize=(12,18))



for i in range(len(numerical.columns)):

   fig.add_subplot(9,4,i+1)

   sns.distplot(numerical.iloc[:,i].dropna(), kde=False)

   plt.xlabel(numerical.columns[i])



plt.tight_layout()
train.columns[train.dtypes == 'object']
categorical = train.select_dtypes(include='object').copy()

categorical
categorical.shape
#Repartition of values for each categorical features.

for elt in categorical.columns:

    print("{} -> {}".format(elt, categorical[elt].unique()), end='\n\n')
#Histogram.

fig = plt.figure(figsize=(15,25))



for i in range(len(categorical.columns)):

   fig.add_subplot(9,5,i+1)

   ax = sns.countplot(categorical.iloc[:,i].dropna())

   ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)

   plt.xlabel(categorical.columns[i])



plt.tight_layout()
train['Utilities'].value_counts()
#plt.title('Distribution of SalePrice')

#sns.distplot(train['SalePrice'])



#fig = plt.figure()

#stats.probplot(train['SalePrice'], plot=plt)



print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
f = plt.figure(figsize=(12,20))



for i in range(len(numerical.columns)):

   f.add_subplot(9, 4, i+1)

   sns.scatterplot(numerical.iloc[:,i], train['SalePrice'])

   

plt.tight_layout()
correlation_mat = train.corr()



f, ax = plt.subplots(figsize=(12,9))

plt.title('Correlation of numerical attributes', size=16)

sns.heatmap(correlation_mat, vmin=0.2, vmax=0.8, square=True, cmap='BuPu')

plt.show()
correlation_mat['SalePrice'].sort_values(ascending=False)
#Numerical

percentage_missing = numerical.isna().sum() / len(train) * 100

percentage_missing.sort_values(ascending=False)
#Categorical

percentage_missing = categorical.isna().sum() / len(train) * 100

percentage_missing.sort_values(ascending=False)
train.PoolQC.value_counts()
#Remove outliers in LotFrontage.

train.drop(train[train['LotFrontage'] > 200].index, inplace=True)



#Remove outliers in LotArea.

train.drop(train[train['LotArea'] > 100000].index, inplace=True)



#Remove outliers in BsmtFinSF1.

train.drop(train[train['BsmtFinSF1'] > 4000].index, inplace=True)



#No need to remove outliers in TotalBsmtSF because we will delete this feature further (high-correlated).



#Remove outliers in 1stFlrSF.

train.drop(train[train['1stFlrSF'] > 4000].index, inplace=True)



#Remove outliers in GrLivArea.

train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index, inplace=True)



#Remove outliers in LowQualFinSF.

train.drop(train[train['LowQualFinSF'] > 550].index, inplace=True)





#fig = plt.figure()

plt.title('Probability plot with kurtosis fixed')

stats.probplot(train['SalePrice'], plot=plt)

print("Kurtosis: %f" % train['SalePrice'].kurt())
train['SalePrice'] = np.log(train['SalePrice'])
fig = plt.figure()

plt.title('Distribution of SalePrice without skewness')

sns.distplot(train['SalePrice'])

print("Skewness: %f" % train['SalePrice'].skew())
#Concat train/test set.

all_data = pd.concat([train.drop('SalePrice', axis=1), test], sort=False)
all_data.shape
#Numerical missing values.

all_data[numerical.columns].isna().sum().sort_values(ascending=False).head(15)
tmp = ['LotFrontage', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF']



all_data[tmp] = all_data[tmp].fillna(0)

all_data[numerical.columns] = all_data[numerical.columns].fillna(all_data[numerical.columns].mean())
#Categorical missing values.

all_data[categorical.columns].isna().sum().sort_values(ascending=False).head(15)
tmp = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish',

       'GarageType', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

all_data[tmp] = all_data[tmp].fillna('None')



all_data[categorical.columns] = all_data[categorical.columns].fillna(all_data[categorical.columns].mode().iloc[0, :])
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['TotalBathroom'] = all_data['FullBath'] + all_data['HalfBath']

all_data.drop(['FullBath', 'HalfBath'],axis=1,inplace=True)
features_to_drop = ['GarageArea', 'TotalBsmtSF', 'GarageYrBlt', 'TotRmsAbvGrd']

all_data.drop(columns=features_to_drop, inplace=True)
all_data.drop(columns='Utilities', inplace=True)
all_data = pd.get_dummies(all_data)
all_data.shape
nTrain = train.SalePrice.shape[0]
train_transf = all_data[:nTrain]

test_transf = all_data[nTrain:]
#Split dataset into training and validation set.

#X_train, X_val, y_train, y_val = train_test_split(train_transf, train['SalePrice'], random_state=1)
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold

from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
lin_reg = LinearRegression()

lasso = Lasso(alpha =0.0005, random_state=1)

ridge = Ridge(alpha =0.0005, random_state=1)

Enet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

model_lgb = LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



base_models = [lin_reg, lasso, ridge, Enet, GBoost, model_xgb, model_lgb]
# header_list = ['y_lin_reg', 'y_lasso', 'y_ridge', 'y_Enet', 'y_GBoost', 'y_model_xgb', 'y_model_lgb', 'y_cb']

header_list = ['y_lin_reg', 'y_lasso', 'y_ridge', 'y_Enet', 'y_GBoost', 'y_model_xgb', 'y_model_lgb']

new_train_dataset = pd.DataFrame(columns=header_list)

new_test_dataset = pd.DataFrame()

#Enable us to pick the meta model.

mae_compare = pd.Series(index=header_list)
kfold = KFold(n_splits=6, random_state=42)



#For each model.

for model, header in zip(base_models, header_list):

    #Fit 80% of the training set (train_index) and predict on the remaining 20% (oof_index). 

    for train_index, oof_index in kfold.split(train_transf, train['SalePrice']):

        X_train, X_val = train_transf.iloc[train_index, :], train_transf.iloc[oof_index, :]

        y_train, y_val = train.SalePrice.iloc[train_index], train.SalePrice.iloc[oof_index]

    

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        new_train_dataset[header] = y_pred

        

        mae_compare[header] = mean_absolute_error(np.exp(y_val), np.exp(y_pred))

    

    #Create new_test_set at the same time.

    print(header)

    new_test_dataset[header] =  model.predict(test_transf)



# #Add y_val to new_train_dataset.

# #If we don't drop the ID, we will get NaN if ID doesn't match with new_train_dataset index.

new_train_dataset['y_val'] = y_val.reset_index(drop=True)
#Pick the meta-model.

mae_compare.sort_values(ascending=True)
#Train meta-model on new_train_dataset.

lasso.fit(new_train_dataset.iloc[:, :-1], new_train_dataset.iloc[:, -1])
#Apply train meta-model to new_test_dataset.

y_meta_pred = lasso.predict(new_test_dataset)
print('{}: {}'.format('train_transf',train_transf.shape))

print('{}: {}'.format('test_transf',test_transf.shape))

print('{}: {}'.format('X_train',X_val.shape))

print('{}: {}'.format('y_train',y_val.shape))

print('{}: {}'.format('X_val',X_val.shape))

print('{}: {}'.format('y_val',y_val.shape))

print('{}: {}'.format('new_train_dataset',new_train_dataset.shape))

print('{}: {}'.format('new_test_dataset',new_test_dataset.shape))

print('{}: {}'.format('y_meta_pred',y_meta_pred.shape))
# #Submission

# output = pd.DataFrame({'Id': test.index,

#                        'SalePrice': (np.exp(y_meta_pred))})

# output.to_csv('stacking_submission.csv', index=False)