#import some necessary librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points





from subprocess import check_output

print(check_output(["ls", "../input/house-prices-advanced-regression-techniques"]).decode("utf8")) #check the files available in the directory



import xgboost as xgb

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
def delete_id(train, test):

    train_ID = train['Id']

    test_ID = test['Id']



    #Now drop the  'Id' colum since it's unnecessary for  the prediction process.

    train.drop("Id", axis = 1, inplace = True)

    test.drop("Id", axis = 1, inplace = True)

    

    return train_ID,test_ID,train,test



def delete_outliers(train):

    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

    return train



def log_target(train):

    train["SalePrice"] = np.log1p(train["SalePrice"])

    return train



def construct_all_data(train, test):

    ntrain = train.shape[0]

    ntest = test.shape[0]

    y_train = train.SalePrice.values

    all_data = pd.concat((train, test)).reset_index(drop=True)

    all_data.drop(['SalePrice'], axis=1, inplace=True)

    return ntrain,ntest,all_data



def data_filling(all_data):

    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

    all_data["Alley"] = all_data["Alley"].fillna("None")

    all_data["Fence"] = all_data["Fence"].fillna("None")

    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

        all_data[col] = all_data[col].fillna('None')

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

        all_data[col] = all_data[col].fillna(0)

    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

        all_data[col] = all_data[col].fillna(0)

    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

    all_data = all_data.drop(['Utilities'], axis=1)

    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

    return all_data



def change_data_type(all_data):

    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

    all_data['OverallCond'] = all_data['OverallCond'].astype(str)

    all_data['YrSold'] = all_data['YrSold'].astype(str)

    all_data['MoSold'] = all_data['MoSold'].astype(str)

    return all_data



def label_encode(all_data):

    

    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

    # process columns, apply LabelEncoder to categorical features

    for c in cols:

        lbl = LabelEncoder() 

        lbl.fit(list(all_data[c].values)) 

        all_data[c] = lbl.transform(list(all_data[c].values))

    return all_data



def add_features(all_data):

    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    return all_data

    

def fix_skew_features(all_data):

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



    # Check the skew of all numerical features

    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

    #print("\nSkew in numerical features: \n")

    skewness = pd.DataFrame({'Skew' :skewed_feats})

    

    skewness = skewness[abs(skewness) > 0.75]

    #print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



    from scipy.special import boxcox1p

    skewed_features = skewness.index

    lam = 0.15

    for feat in skewed_features:

        #all_data[feat] += 1

        all_data[feat] = boxcox1p(all_data[feat], lam)

    return all_data



def get_dummys_data(all_data):

    all_data = pd.get_dummies(all_data)

    return all_data



def get_train_test(all_data,ntrain):

    train = all_data[:ntrain]

    test = all_data[ntrain:]

    return train,test



def rmsle_cv(model,n_folds = 5):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID,test_ID,train,test = delete_id(train, test)

train = delete_outliers(train)

train = log_target(train)

y_train = train.SalePrice.values

ntrain,ntest,all_data = construct_all_data(train, test)

all_data = data_filling(all_data)

all_data = change_data_type(all_data)

all_data = label_encode(all_data)

all_data = add_features(all_data)

all_data = fix_skew_features(all_data)

all_data = get_dummys_data(all_data)

train,test = get_train_test(all_data,ntrain)

model = xgb.XGBRegressor()

model.fit(train, y_train)

xgb_train_pred = model.predict(train)

print(rmsle(y_train, xgb_train_pred))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID,test_ID,train,test = delete_id(train, test)

train = log_target(train)

y_train = train.SalePrice.values

ntrain,ntest,all_data = construct_all_data(train, test)

all_data = data_filling(all_data)

all_data = change_data_type(all_data)

all_data = label_encode(all_data)

all_data = add_features(all_data)

all_data = fix_skew_features(all_data)

all_data = get_dummys_data(all_data)

train,test = get_train_test(all_data,ntrain)

model = xgb.XGBRegressor()

model.fit(train, y_train)

xgb_train_pred = model.predict(train)

print(rmsle(y_train, xgb_train_pred))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID,test_ID,train,test = delete_id(train, test)

train = delete_outliers(train)

y_train = train.SalePrice.values

ntrain,ntest,all_data = construct_all_data(train, test)

all_data = data_filling(all_data)

all_data = change_data_type(all_data)

all_data = label_encode(all_data)

all_data = add_features(all_data)

all_data = fix_skew_features(all_data)

all_data = get_dummys_data(all_data)

train,test = get_train_test(all_data,ntrain)

model = xgb.XGBRegressor()

model.fit(train, y_train)

xgb_train_pred = model.predict(train)

print(rmsle(np.log1p(y_train), np.log1p(xgb_train_pred)))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID,test_ID,train,test = delete_id(train, test)

train = delete_outliers(train)

train = log_target(train)

y_train = train.SalePrice.values

ntrain,ntest,all_data = construct_all_data(train, test)

all_data = all_data.fillna(0)

all_data = change_data_type(all_data)

all_data = label_encode(all_data)

all_data = add_features(all_data)

all_data = fix_skew_features(all_data)

all_data = get_dummys_data(all_data)

train,test = get_train_test(all_data,ntrain)

model = xgb.XGBRegressor()

model.fit(train, y_train)

xgb_train_pred = model.predict(train)

print(rmsle(y_train, xgb_train_pred))
def temp_label_encode(all_data):

    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir')

    # process columns, apply LabelEncoder to categorical features

    for c in cols:

        lbl = LabelEncoder() 

        lbl.fit(list(all_data[c].values)) 

        all_data[c] = lbl.transform(list(all_data[c].values))

    return all_data



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID,test_ID,train,test = delete_id(train, test)

train = delete_outliers(train)

train = log_target(train)

y_train = train.SalePrice.values

ntrain,ntest,all_data = construct_all_data(train, test)

all_data = data_filling(all_data)

all_data = temp_label_encode(all_data)

all_data = add_features(all_data)

all_data = fix_skew_features(all_data)

all_data = get_dummys_data(all_data)

train,test = get_train_test(all_data,ntrain)

model = xgb.XGBRegressor()

model.fit(train, y_train)

xgb_train_pred = model.predict(train)

print(rmsle(y_train, xgb_train_pred))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID,test_ID,train,test = delete_id(train, test)

train = delete_outliers(train)

train = log_target(train)

y_train = train.SalePrice.values

ntrain,ntest,all_data = construct_all_data(train, test)

all_data = data_filling(all_data)

all_data = change_data_type(all_data)

all_data = label_encode(all_data)

all_data = fix_skew_features(all_data)

all_data = get_dummys_data(all_data)

train,test = get_train_test(all_data,ntrain)

model = xgb.XGBRegressor()

model.fit(train, y_train)

xgb_train_pred = model.predict(train)

print(rmsle(y_train, xgb_train_pred))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID,test_ID,train,test = delete_id(train, test)

train = delete_outliers(train)

train = log_target(train)

y_train = train.SalePrice.values

ntrain,ntest,all_data = construct_all_data(train, test)

all_data = data_filling(all_data)

all_data = change_data_type(all_data)

all_data = label_encode(all_data)

all_data = add_features(all_data)

all_data = get_dummys_data(all_data)

train,test = get_train_test(all_data,ntrain)

model = xgb.XGBRegressor()

model.fit(train, y_train)

xgb_train_pred = model.predict(train)

print(rmsle(y_train, xgb_train_pred))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID,test_ID,train,test = delete_id(train, test)

train = delete_outliers(train)

train = log_target(train)

y_train = train.SalePrice.values

ntrain,ntest,all_data = construct_all_data(train, test)

all_data = data_filling(all_data)

all_data = change_data_type(all_data)

all_data = label_encode(all_data)

all_data = add_features(all_data)

all_data = fix_skew_features(all_data)

all_data = get_dummys_data(all_data)

train,test = get_train_test(all_data,ntrain)

model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

model.fit(train, y_train)

xgb_train_pred = model.predict(train)

print(rmsle(y_train, xgb_train_pred))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID,test_ID,train,test = delete_id(train, test)

train = delete_outliers(train)

train = log_target(train)

y_train = train.SalePrice.values

ntrain,ntest,all_data = construct_all_data(train, test)

all_data = data_filling(all_data)

all_data = change_data_type(all_data)

all_data = label_encode(all_data)

all_data = add_features(all_data)

all_data = fix_skew_features(all_data)

all_data = get_dummys_data(all_data)

train,test = get_train_test(all_data,ntrain)



xgb_model = xgb.XGBRegressor()

xgb_model.fit(train, y_train)

xgb_train_pred = xgb_model.predict(train)



lgb_model = lgb.LGBMRegressor()

lgb_model.fit(train, y_train)

lgb_train_pred = lgb_model.predict(train)



print(rmsle(y_train, xgb_train_pred*0.5 + lgb_train_pred*0.5))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID,test_ID,train,test = delete_id(train, test)

train = delete_outliers(train)

train = log_target(train)

y_train = train.SalePrice.values

ntrain,ntest,all_data = construct_all_data(train, test)

all_data = data_filling(all_data)

all_data = temp_label_encode(all_data)

all_data = add_features(all_data)

all_data = fix_skew_features(all_data)

all_data = get_dummys_data(all_data)

train,test = get_train_test(all_data,ntrain)



xgb_model = xgb.XGBRegressor()

xgb_model.fit(train, y_train)

xgb_train_pred = xgb_model.predict(train)



lgb_model = lgb.LGBMRegressor()

lgb_model.fit(train, y_train)

lgb_train_pred = lgb_model.predict(train)



print(rmsle(y_train, xgb_train_pred*0.5 + lgb_train_pred*0.5))



xgb_pred = np.expm1(xgb_model.predict(test))

lgb_pred = np.expm1(lgb_model.predict(test))

ensemble = xgb_pred*0.5 + lgb_pred*0.5



sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = ensemble

sub.to_csv('submission.csv',index=False)