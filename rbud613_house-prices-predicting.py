import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print("train shape : ",train.shape,"  test shape : ",test.shape)
train1 = train[train.columns[train.isnull().sum()!=0]]

test1 = test[test.columns[test.isnull().sum()!=0]]

na_prop1 = (train1.isnull().sum() / train1.shape[0]).sort_values()

na_prop2 = (test1.isnull().sum() / test1.shape[0]).sort_values()

plt.figure(figsize=(10,8))

sns.set_style('whitegrid')

plt.subplot(211)

na_prop1.plot.barh(color='blue')

plt.title('Missing values(train set)', weight='bold')

plt.subplot(212)

na_prop2.plot.barh(color='blue')

plt.title('Missing values(test set)', weight='bold' )
Lot_na = na_prop1.sort_values(ascending=False).index[:6]

print(train.drop(Lot_na,axis=1).dropna(axis=0).shape)

print(test.drop(Lot_na,axis=1).dropna(axis=0).shape)
plt.figure(figsize = (8,3))

plt.subplot(121)

sns.distplot(train['SalePrice'])

plt.subplot(122)

stats.probplot(train['SalePrice'],plot=plt)
plt.figure(figsize = (8,3))

plt.subplot(121)

sns.distplot(np.log(train['SalePrice']))

plt.subplot(122)

stats.probplot(np.log(train['SalePrice']),plot=plt)
plt.figure(figsize = (8,3))

plt.subplot(121)

sns.residplot(x = train.Id, y = train.SalePrice)

plt.subplot(122)

sns.residplot(x = train.Id, y = np.log(train.SalePrice))
plt.figure(figsize=(15,5))

sns.scatterplot(x='YearBuilt', y="SalePrice", data=train)
train1 = train.drop('SalePrice',axis=1).select_dtypes(exclude='object')

train2 = train.select_dtypes(include='object')

test1 = test.select_dtypes(exclude='object')

print("Numeric column number : ", train1.shape[1])

print("Categorical column number : ", train2.shape[1])
plt.rc('axes',labelsize=9)

plt.rc('xtick',labelsize=10)

plt.rc('ytick',labelsize=10)

plt.figure(figsize = (15,18))

plt.subplots_adjust(hspace = 0.5, wspace = 0.3)

for i,col in enumerate(train1.columns[1:]):

    plt.subplot(6,6,i+1)

    sns.distplot(train1[col],kde=False)
plt.figure(figsize = (15,18))

plt.subplots_adjust(hspace = 0.5, wspace = 0.3)

for i,col in enumerate(test1.columns[1:]):

    plt.subplot(6,6,i+1)

    sns.distplot(test1[col],kde=False)
pd.concat([train1.describe().T[["min","max"]] ,test1.describe().T[["min","max"]]],axis=1, sort=False)
plt.figure(figsize = (20,20))

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

for i,col in enumerate(train1.columns[1:]):

    plt.subplot(6,6,i+1)

    sns.scatterplot(x=col, y="SalePrice", data=train)
skew_list = []

for col in train1.columns:

    median = train1[col].median()

    skew = stats.skew(train1[col].fillna(median))

    skew_list.append(skew)

skew_table = pd.DataFrame(skew_list, index = train1.columns, columns=["skew"])



plt.figure(figsize=(13,4))

sns.barplot(skew_table.index, skew_table["skew"])

plt.xticks(rotation=90)
plt.style.use('seaborn-darkgrid')

plt.figure(figsize = (18,25))

plt.subplots_adjust(hspace = 0.7, wspace = 0.5)

for i,col in enumerate(train2.columns[1:]):

    plt.subplot(9,5,i+1)

    sns.boxplot(train2[col],train['SalePrice'],linewidth=2)

    plt.xticks(rotation=90)
for col in train1.columns:

    print('{} : {}'.format(col,round(train[col].var()/train[col].mean(),3),2))
plt.figure(figsize=(4,13))

sns.heatmap(train.corr(method="spearman")[["SalePrice"]].sort_values(by = "SalePrice", ascending = False)[1:],annot=True,cmap = "Blues")
plt.figure(figsize = (15,15))

sns.heatmap(train.corr(method="spearman"),cmap = 'Blues',annot=True,fmt='.1f')
low_cor = train.corr(method="spearman")['SalePrice']

low_cor = low_cor[abs(low_cor)<0.1]

plt.rc('axes',labelsize=9)

plt.rc('xtick',labelsize=10)

plt.rc('ytick',labelsize=10)

plt.figure(figsize = (20,8))

plt.subplots_adjust(hspace = 0.5, wspace = 0.3)

for i,col in enumerate(low_cor.index[1:]):

    plt.subplot(2,5,i+1)

    sns.scatterplot(train[col],train['SalePrice'])
drop_col = ["PoolQC", "MiscFeature", "Alley", "Fence", "Street", "Utilities"]



test.loc[test['GarageYrBlt'] == 2207,'GarageYrBlt'] = 2007



for df in [train, test]:

    df.drop(drop_col, axis=1, inplace=True)

    for col in ["FireplaceQu","GarageType", "GarageFinish", "GarageQual", "GarageCond",

                "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",'MasVnrType']:

        df[col] = df[col].fillna("None")

    

    for col in ['GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']:

        df[col] = df[col].fillna(0)
for df in [train, test]:

    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

    df["Functional"] = df["Functional"].fillna(df['SaleType'].mode()[0])

    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])

    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])

    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])

    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

    df["LotFrontage"].interpolate(method='linear',inplace=True)

    df['MoSold'] = df['MoSold'].astype(str)

    df['YrSold'] = df['YrSold'].astype(str)

    df['GarageYrBlt'] = df['GarageYrBlt'].astype(str)

    df['BsmtUnfSF'] = np.log1p(df['BsmtUnfSF'])
train1 = train[train.columns[train.isnull().sum()!=0]]

test1 = test[test.columns[test.isnull().sum()!=0]]

na_prop1 = (train1.isnull().sum() / train1.shape[0]).sort_values()

na_prop2 = (test1.isnull().sum() / test1.shape[0]).sort_values()

print(na_prop1)

print(na_prop2)
train = train.drop(train[train.LotArea > 100000].index)

train = train.drop(train[(train.OverallQual==10) & (train.SalePrice<200000)].index)

train = train.drop(train[(train.OverallCond==6) & (train.SalePrice > 700000)].index)

train = train.drop(train[(train.GrLivArea>4000) & (train.SalePrice<200000)].index)

train = train.drop(train[(train.YearBuilt<1900) & (train.SalePrice>400000)].index)

train = train.drop(train[(train.YearBuilt>2000) & (train.SalePrice<100000)].index)

train = train.drop(train[(train.MasVnrArea==0) & (train.SalePrice>650000)].index)

train = train.drop(train[(train.OpenPorchSF>=500) & (train.SalePrice<50000)].index)

train = train.drop(train[(train.BsmtHalfBath==1) & (train.SalePrice>700000)].index)

train = train.drop(train[(train.YrSold=="2007") & (train.SalePrice>700000)].index)

train = train.drop(train[(train.BedroomAbvGr==8)].index)

train = train.drop(train[(train.LotFrontage>=300)].index)

train.reset_index(drop=True,inplace=True)

train["SalePrice"] = np.log(train["SalePrice"])

print(train.shape)
test.loc[(test['YearBuilt'] > test['YearRemodAdd']), 'YearRemodAdd'] = test.loc[(test['YearBuilt'] > test['YearRemodAdd']), 'YearBuilt']

for df in [train, test]:

    df['Remod-Bulit'] = df['YearRemodAdd'] - df['YearBuilt']

    df['Total_Bathrooms'] = df['FullBath'] + df['HalfBath'] + df['BsmtFullBath'] + df['BsmtHalfBath']

    df['Total_SF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['Total_Square_Feet'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['Total_Porch'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

    df['ispool'] = df['PoolArea'].apply(lambda x: "1" if x > 0 else "0")

    df['isgarage'] = df['GarageArea'].apply(lambda x: "1" if x > 0 else "0")

    df.drop('PoolArea',axis=1, inplace = True)
train1 = train.drop('SalePrice',axis=1).select_dtypes(exclude='object')

train2 = train.select_dtypes(include='object')

print("Numeric column number : ", train1.shape[1])

print("Categorical column number : ", train2.shape[1])
from sklearn.preprocessing import LabelEncoder

train_y = train["SalePrice"]

all_data = pd.concat([train.drop('SalePrice',axis=1),test],axis=0,sort=False)

train2 = train.select_dtypes(include='object')

for c in train2.columns:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))

train = all_data.iloc[:train.shape[0], :]

test = all_data.iloc[train.shape[0]:, :]
train.drop('Id', axis=1,inplace=True)

test.drop('Id', axis=1,inplace=True)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train, train_y, test_size=0.2, random_state=10)
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

lgb_regressor=lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.02, n_estimators=2310, max_bin=50, bagging_fraction=0.9,bagging_freq=5, bagging_seed=7, 

                                feature_fraction=0.9, feature_fraction_seed=123,n_jobs=-1)

lgb_regressor.fit(x_train,y_train)

y_head=lgb_regressor.predict(x_test)

mean_squared_error(y_test, y_head)
tmp = pd.DataFrame({'Feature': x_train.columns, 'Feature importance': lgb_regressor.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (15,15))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)
from sklearn.model_selection import KFold 

from sklearn.metrics import mean_squared_error

def lgb_cv(n_estimators, reg_alpha, reg_lambda, min_split_gain, min_child_weight,min_child_samples, colsample_bytree, x_data=None, y_data=None, n_splits=5, output='score'):

    score = 0

    kf = KFold(n_splits=n_splits)

    models = []

    for train_index, valid_index in kf.split(x_data):

        x_train, y_train = x_data.iloc[train_index], y_data[train_index]

        x_valid, y_valid = x_data.iloc[valid_index], y_data[valid_index]

        

        model = lgb.LGBMRegressor(

            num_leaves = 4, 

            learning_rate = 0.01, 

            n_estimators = int(n_estimators), 

            reg_alpha = reg_alpha, 

            reg_lambda = reg_lambda,

            min_split_gain= min_split_gain,

            min_child_weight = min_child_weight,

            min_child_samples = int(min_child_samples),

            colsample_bytree = np.clip(colsample_bytree, 0, 1), 

        )

        

        model.fit(x_train, y_train)

        models.append(model)

        

        pred = model.predict(x_valid)

        true = y_valid

        score -= mean_squared_error(true, pred)/n_splits

    

    if output == 'score':

        return score

    if output == 'model':

        return models
from functools import partial 

from bayes_opt import BayesianOptimization

func_fixed = partial(lgb_cv, x_data=train, y_data=train_y, n_splits=5, output='score')

lgbBO = BayesianOptimization(

    func_fixed, 

    {     

        'n_estimators': (1000, 3000),                        

        'reg_alpha': (0.0001, 1),       

        'reg_lambda': (0.0001, 1), 

        'min_split_gain' : (0.001, 0.1),

        'min_child_weight' : (0.001, 0.1),

        'min_child_samples' : (10,25),

        'colsample_bytree': (0.85, 1.0),

    }, 

    random_state=4321            

)

lgbBO.maximize(init_points=5, n_iter=20)
params = lgbBO.max['params']

lgb_models = lgb_cv(

    params['n_estimators'], 

    params['reg_alpha'], 

    params['reg_lambda'], 

    params['min_split_gain'], 

    params['min_child_weight'],

    params['min_child_samples'],

    params['colsample_bytree'],

    x_data=train, y_data=train_y, n_splits=5, output='model')
from sklearn.ensemble import GradientBoostingRegressor

gb_regressor = GradientBoostingRegressor(n_estimators=1992, learning_rate=0.03, max_depth=3, max_features='sqrt', min_samples_leaf=15, min_samples_split=8, loss='huber', random_state =42)

gb_regressor.fit(x_train,y_train)

y_head=gb_regressor.predict(x_test)

mean_squared_error(y_test, y_head)
from xgboost import XGBRegressor

xgb_regressor = XGBRegressor(learning_rate=0.02,n_estimators = 2400,max_depth=3, min_child_weight=0.01,gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,objective='reg:linear', nthread=-1, scale_pos_weight=1, seed=27,reg_alpha=0.00006)

xgb_regressor.fit(x_train,y_train)

y_head=xgb_regressor.predict(x_test)

mean_squared_error(y_test, y_head)
from mlxtend.regressor import StackingCVRegressor

stack_regressor = StackingCVRegressor(regressors=(gb_regressor, lgb_regressor, xgb_regressor),meta_regressor=gb_regressor, use_features_in_secondary=True)

stack_regressor.fit(np.array(x_train),np.array(y_train))

y_head=stack_regressor.predict(np.array(x_test))

mean_squared_error(y_test, y_head)
final_pred1 = lgb_regressor.predict(test)

preds = []

for model in lgb_models:

    pred = model.predict(test)

    preds.append(pred)

final_pred2 = np.mean(preds, axis=0)

final_pred3 = gb_regressor.predict(test)

final_pred4 = xgb_regressor.predict(test)

final_pred5 = stack_regressor.predict(np.array(test))
y_pred = np.floor(np.exp(0.2*final_pred1+0.2*final_pred2 + 0.2 * final_pred3 + 0.2 * final_pred4  + 0.2 * final_pred3))
submit = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submit["SalePrice"] = y_pred

submit.to_csv('submit.csv', index=False)

submit.head()