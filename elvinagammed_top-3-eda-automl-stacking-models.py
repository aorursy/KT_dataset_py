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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train
# Analysis
# pip install pandas-profiling
from pandas_profiling import ProfileReport
# pr = train.profile_report()
# filling CPU
# pr.to_widgets()
# Cleaning
!pip install dabl
import dabl
train_clean = dabl.clean(train, verbose=0)
types = dabl.detect_types(train_clean)
print(types) 
# Not Normally distributed
dabl.plot(train, 'SalePrice')
# !pip install missingno
import missingno as msno
msno.matrix(train)
msno.dendrogram(train)
msno.bar(train)
msno.matrix(train.sample(100))
msno.heatmap(train)
# !pip install autoviz
!pip install datasist
import datasist as ds
ds.structdata.check_train_test_set(train, test, index=None, col=None)
ds.structdata.describe(train)

numerical_feats = ds.structdata.get_num_feats(train)
ds.structdata.detect_outliers(train,80,numerical_feats)
ds.structdata.display_missing(train)['missing_percent'].sort_values(ascending=False)
all_data, ntrain, ntest = ds.structdata.join_train_and_test(train, test)
print("New size of combined data {}".format(all_data.shape))
print("Old size of train data: {}".format(ntrain))
print("Old size of test data: {}".format(ntest))

#later splitting after transformations
train_new = all_data[:ntrain]
test_new = all_data[ntrain:]
new_train_df = ds.feature_engineering.drop_missing(train_new,  
                                                    percent=7.0)
# ds.structdata.display_missing(new_train_df)
ds.structdata.display_missing(new_train_df)['missing_percent'].sort_values(ascending=False)
ds.feature_engineering.drop_redundant(new_train_df)
df = ds.feature_engineering.fill_missing_cats(new_train_df)
ds.structdata.display_missing(df)['missing_counts'].sort_values()
df = ds.feature_engineering.fill_missing_num(new_train_df)
ds.structdata.display_missing(df)
df = ds.feature_engineering.fill_missing_num(df)
df = ds.feature_engineering.log_transform(df,columns=['Id'])
df
features = ['OverallQual' , 'GrLivArea' , 'TotalBsmtSF' , 'BsmtFinSF1' ,
            '2ndFlrSF'    , 'GarageArea', '1stFlrSF'    , 'YearBuilt'  ]
X_train = df[features]
y_train = df["SalePrice"]
X_test  = test[features]
X_train      = X_train.fillna(X_train.mean())
X_test       = X_test.fillna(X_test.mean())
from rgf.sklearn import RGFRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split

estimators =  [('xgb',xgb.XGBRegressor(n_estimators  = 750,learning_rate = 0.02, max_depth = 4)),
               ('cat',CatBoostRegressor(loss_function='RMSE', verbose=False)),
               ('RGF',RGFRegressor(max_leaf=300, algorithm="RGF_Sib", test_interval=100, loss="LS"))]

ensemble = StackingRegressor(estimators      =  estimators,
                             final_estimator =  RandomForestRegressor())

# Fit ensemble using cross-validation
X_tr, X_te, y_tr, y_te = train_test_split(X_train,y_train)
ensemble.fit(X_tr, y_tr).score(X_te, y_te)

# Prediction
predictions = ensemble.predict(X_test)
out = pd.DataFrame({"Id": test.Id, "SalePrice": predictions})
out
out.to_csv("sub.csv", index=False)
# a Better Score
train.corr()['SalePrice'].nlargest(10)
# Outlier
plt.scatter(train['GrLivArea'], train['SalePrice'])
train = train[train['GrLivArea']<4500]
plt.scatter(train['GrLivArea'], train['SalePrice'])
# not Normal Distributed
train['SalePrice'].plot(kind="kde")
train['SalePrice'] = np.log1p(train['SalePrice'])
train['SalePrice'].plot(kind="kde")
train = ds.feature_engineering.drop_missing(train, percent=7.0)
# Electrical has only one missing
train.isna().sum().sort_values().tail(20)
train[train['Electrical'].isna()]
# Remove it
train = train[~train['Electrical'].isna()]
train[train['MasVnrArea'].isna()]
train[train['MasVnrType'].isna()]
# drop 37 NaNs
train = train[~train['MasVnrArea'].isna()]
# drop 37 NaNs
train = train[~train['MasVnrType'].isna()]
# year month to categorical
cols = ["MSSubClass", "YrSold", 'MoSold']
train[cols] = train[cols].astype(str)
test[cols] = test[cols].astype(str)
train['YrSold']
# Collinearity
train.corr()['GarageArea']['GarageCars']
# should remove one
# not working
# ds.feature_engineering.drop_redundant(train)


# Select upper triangle of correlation matrix
upper = train.corr().abs().where(np.triu(np.ones(train.corr().shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
to_drop = to_drop[:4]
train = train.drop(train[to_drop],axis=1)
test = test.drop(test[to_drop],axis=1)
def fill_missing_data(df):
    df_data = df.copy()
    categoricals = []
    for cname,dtype in df_data.dtypes.items():
        if dtype == 'object':
            categoricals.append(cname)
    # Fill 'None' for the Categorical attribute
    df_data[categoricals] = df_data[categoricals].fillna('None')
    
    for cname in df_data.columns:
        if cname not in categoricals:
            df_data[cname] = df_data[cname].fillna(0) #Fill 0 for the Numeric attribute
    return df_data

train = fill_missing_data(train)
test = fill_missing_data(test)
train['TotalPorchSF'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
test['TotalPorchSF'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']
train['TotalBaths'] = train['BsmtFullBath'] + train['FullBath'] + 0.5*(train['BsmtHalfBath'] + train['HalfBath'])
test['TotalBaths'] = test['BsmtFullBath'] + test['FullBath'] + 0.5*(test['BsmtHalfBath'] + test['HalfBath'])
train['TotalAreaSF'] = train['TotalBsmtSF'] + train['GrLivArea']
test['TotalAreaSF'] = test['TotalBsmtSF'] + test['GrLivArea']
train['Age'] = train['YrSold'].astype('int64') - train['YearBuilt']
test['Age'] = test['YrSold'].astype('int64') - test['YearBuilt']
# one hot encode
def feature_engineering(df):
    df_data = df.copy()
    
    feature = {
        'categorical':{
            'MSSubClass': ['20', '30', '40', '45', '50', '60', '70', '75', '80', '85', '90', '120', '150', '160', '180', '190'],
            'MSZoning': ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
            'Alley': ['Grvl', 'Pave', 'None'],
            'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],
            'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
            'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                            'Names', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'],
            'Condition1': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
            'Condition2': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
            'BldgType': ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'],
            'HouseStyle': ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
            'RoofStyle': ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
            'RoofMatl': ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
            'Exterior1st': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco',
                           'VinylSd', 'Wd Sdng', 'WdShing'],
            'Exterior2nd': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco',
                           'VinylSd', 'Wd Sdng', 'WdShing'],
            'MasVnrType': ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'],
            'Foundation': ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
            'Heating': ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
            'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
            'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
            'GarageType': ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'None'],
            'GarageFinish': ['Fin', 'RFn', 'Unf', 'None'],
            'PavedDrive': ['Y', 'P', 'N'],
            'MiscFeature': ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'None'],
            'MoSold': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            'YrSold': ['2006', '2007', '2008', '2009', '2010'],
            'SaleType': ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'],
            'SaleCondition': ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
        },
        'binary': {
            'Street': ['Pave', 'Grvl'],
            'CentralAir': ['Y', 'N']          
        },
        'ordinal': {
            'LotShape': ['None', 'IR3', 'IR2', 'IR1', 'Reg'],
            'Utilities': ['None', 'NoSeWa', 'NoSewr', 'AllPub'],
            'LandSlope': ['None', 'Sev', 'Mod', 'Gtl'],
            'ExterQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'ExterCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
            'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'HeatingQC': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'KitchenQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'Fence': ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
            'PoolQC': ['None', 'Fa', 'Ta', 'Gd', 'Ex']
        },
    }
    
    selected = []
    for cname in df_data.columns:
        if cname in feature['binary']: # Convert the binary attributes to 0/1
            default_value = feature['binary'][cname][0]
            feature_name = cname + "_is_" + default_value
            selected.append(feature_name)
            df_data[feature_name] = df_data[cname].apply(lambda x: int(x == default_value))
        elif cname in feature['categorical']: # Convert Categorical attributes into One-hot vector
            values = feature['categorical'][cname]
            for val in values:
                try:
                    new_name = "{}_{}".format(cname, val)

                    selected.append(new_name)
                    df_data[new_name] = df_data[cname].apply(lambda x: int(x == val))
                except Exception as err:
                    print("One-hot encoding for {}_{}. Error: {}".format(cname, val, err))
        elif cname in feature['ordinal']: # Convert the Ordinal attributes to a number
            new_name = cname + "_ordinal"
            selected.append(new_name)
            df_data[new_name] = df_data[cname].apply(lambda x: int(feature['ordinal'][cname].index(x)))
        else: # The remaining attributes are numeric so they remain the same
#             print(cname)
            selected.append(cname)
            
    return df_data[selected]
train = feature_engineering(train)
test = feature_engineering(test)
train
# remove all zeros columns
# df=df[[i for i in df if len(set(df[i]))>1]]
for col in train.columns:    
    if any(train[col]) == False:
            train.drop([col], axis=1, inplace=True)
            test.drop([col], axis=1, inplace=True)
id = test['Id']
id.shape
y = train['SalePrice']
y.shape
test['Id'].shape
df['SalePrice']
train.drop(['Id', 'SalePrice'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)

abt = test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train = scaler.fit_transform(train)
# test = scaler.transform(test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.25, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=123)
from sklearn.metrics import mean_squared_error
param_init = {
    "max_depth": 5, # default: 3 only for depthwise
    "n_estimators": 3000, # default: 500
    "learning_rate": 0.01, # default: 0.05
    "subsample": 0.5,
    "colsample_bytree": 0.7,  # default:  1.0
    "min_child_weight": 1.5,
    "reg_alpha": 0.75,
    "reg_lambda": 0.4,
    "seed": 42,
#     "eval_metric": "rmse"
}
import xgboost
xgb_model = xgboost.XGBRegressor(**param_init)
param_fit = {
    "eval_metric": "rmse",
    "early_stopping_rounds": 500, # default: 100
    "verbose": 200,
    "eval_set": [(X_val, y_val)]
}
xgb_model = xgb_model.fit(X_train, y_train, **param_fit)

y_pred_xgb = xgb_model.predict(X_test)
SalePrice_pred
mean_squared_error(y_test, y_pred_xgb, squared=False)

SalePrice_pred = np.exp(y_pred_xgb)
y_pred_xgb.shape
# Id=pd.Series(range(1461,2920))
# submission = pd.DataFrame({'Id': Id, 'SalePrice': SalePrice_pred})
# submission.to_csv("sub.csv", index=False)