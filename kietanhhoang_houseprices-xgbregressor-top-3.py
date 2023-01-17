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
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
abs(df_train.corr()['SalePrice']).nlargest(10)
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')
df_train = df_train[df_train["GrLivArea"] < 4500]
sns.distplot(df_train['SalePrice'])
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
from scipy.stats import norm

sns.distplot(df_train['SalePrice'], fit=norm)
n = df_train.shape[0]

for col in df_train.columns:

    missing_pct = sum(df_train[col].isnull())*100.0/n

    if missing_pct > 95.0:

        print('{}: {:0.2f}%'.format(col, missing_pct))
df_train.drop(['PoolQC', 'MiscFeature'], axis=1, inplace=True)

df_test.drop(['PoolQC', 'MiscFeature'], axis=1, inplace=True)
df_train[df_train['Electrical'].isna()]
df_train = df_train[~df_train['Electrical'].isna()]
cols = ["MSSubClass", "YrSold", 'MoSold']

df_train[cols] = df_train[cols].astype(str)

df_test[cols] = df_test[cols].astype(str)
df_train.corr()['GarageArea']['GarageCars']
df_train.corr()['GarageArea']['SalePrice']
df_train.corr()['GarageCars']['SalePrice']
cname = 'GarageYrBlt'

df_train.corr()[cname][abs(df_train.corr()[cname]) > 0.65]
cname = 'GrLivArea'

df_train.corr()[cname][abs(df_train.corr()[cname]) > 0.65]
cname = 'TotalBsmtSF'

df_train.corr()[cname][abs(df_train.corr()[cname]) > 0.65]
col = ['GarageArea','1stFlrSF', '2ndFlrSF','TotRmsAbvGrd', 'GarageYrBlt']

df_train.drop(col, axis=1, inplace=True)

df_test.drop(col, axis=1, inplace=True)
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
df_train = fill_missing_data(df_train)

df_test = fill_missing_data(df_test)
df_train['TotalPorchSF'] = df_train['OpenPorchSF'] + df_train['EnclosedPorch'] + df_train['3SsnPorch'] + df_train['ScreenPorch']

df_test['TotalPorchSF'] = df_test['OpenPorchSF'] + df_test['EnclosedPorch'] + df_test['3SsnPorch'] + df_test['ScreenPorch']
df_train['TotalBaths'] = df_train['BsmtFullBath'] + df_train['FullBath'] + 0.5*(df_train['BsmtHalfBath'] + df_train['HalfBath'])

df_test['TotalBaths'] = df_test['BsmtFullBath'] + df_test['FullBath'] + 0.5*(df_test['BsmtHalfBath'] + df_test['HalfBath'])
df_train['TotalAreaSF'] = df_train['TotalBsmtSF'] + df_train['GrLivArea']

df_test['TotalAreaSF'] = df_test['TotalBsmtSF'] + df_test['GrLivArea']
df_train['Age'] = df_train['YrSold'].astype('int64') - df_train['YearBuilt']

df_test['Age'] = df_test['YrSold'].astype('int64') - df_test['YearBuilt']
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
df_train = feature_engineering(df_train)

df_test = feature_engineering(df_test)

df_train
for col in df_train.columns:

    if any(df_train[col]) == False:

        df_train.drop([col], axis=1, inplace=True)

        df_test.drop([col], axis=1, inplace=True)
ids = df_test['Id']

y = df_train['SalePrice']
df_train.drop(['Id', 'SalePrice'], axis=1, inplace=True)

df_test.drop(['Id'], axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df_train)

train = scaler.transform(df_train)

test = scaler.transform(df_test)
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
mean_squared_error(y_test, y_pred_xgb, squared=False)
from sklearn.tree import DecisionTreeRegressor  



regressorTree = DecisionTreeRegressor(random_state = 0, min_samples_split=2, max_depth=6)  

regressorTree.fit(X_train, y_train) 
y_pred_Tree = regressorTree.predict(X_test)

mean_squared_error(y_test, y_pred_Tree, squared=False)
from sklearn.ensemble import GradientBoostingRegressor



regressorGB = GradientBoostingRegressor(

    max_depth=5,

    n_estimators=10000,

    learning_rate=0.25

)

regressorGB.fit(X_train, y_train)
y_pred_GB = regressorGB.predict(X_test)

mean_squared_error(y_test, y_pred_GB, squared=False)
from sklearn.linear_model import Lasso

regressorLasso = Lasso(alpha=0.0007)

regressorLasso.fit(X_train, y_train)
y_pred_Lasso = regressorLasso.predict(X_test)

mean_squared_error(y_test, y_pred_Lasso, squared=False)
SalePrice_pred = xgb_model.predict(test)

# Because it takes log () to train, it is necessary to take exp () the predicted result

SalePrice_pred = np.exp(SalePrice_pred)
submission = {'Id': ids, 'SalePrice': SalePrice_pred}
df_submission = pd.DataFrame(submission)

df_submission
df_submission.to_csv('submission.csv', index=False)