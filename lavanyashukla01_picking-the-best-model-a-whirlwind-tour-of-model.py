# Please turn the 'Internet' toggle On in the Settings panel to your left, in order to make changes to this kernel.

# Essentials

import numpy as np

import pandas as pd

import datetime

import random



# Plots

import seaborn as sns

import matplotlib.pyplot as plt



# Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor



# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Misc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA



pd.set_option('display.max_columns', None)



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000



import os

print(os.listdir("../input/kernel-files"))



# Set random state for numpy

np.random.seed(42)
# Read in the dataset as a dataframe

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
# Remove the Ids from train and test, as they are unique for each row and hence not useful for the model

train_ID = train['Id']

test_ID = test['Id']

train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)

train.shape, test.shape
# log(1+x) transform

train["SalePrice"] = np.log1p(train["SalePrice"])
# Remove outliers

train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)

train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)

train.reset_index(drop=True, inplace=True)
# Split features and labels

train_labels = train['SalePrice'].reset_index(drop=True)

train_features = train.drop(['SalePrice'], axis=1)

test_features = test



# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset

all_features = pd.concat([train_features, test_features]).reset_index(drop=True)

all_features.shape
# Fill missing values

# determine the threshold for missing values

def percent_missing(df):

    data = pd.DataFrame(df)

    df_cols = list(pd.DataFrame(data))

    dict_x = {}

    for i in range(0, len(df_cols)):

        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})

    

    return dict_x



missing = percent_missing(all_features)

df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)

print('Percent of missing data')

df_miss[0:10]
# Some of the non-numeric predictors are stored as numbers; convert them into strings 

all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)

all_features['YrSold'] = all_features['YrSold'].astype(str)

all_features['MoSold'] = all_features['MoSold'].astype(str)
def handle_missing(features):

    # the data description states that NA refers to typical ('Typ') values

    features['Functional'] = features['Functional'].fillna('Typ')

    # Replace the missing values in each of the columns below with their mode

    features['Electrical'] = features['Electrical'].fillna("SBrkr")

    features['KitchenQual'] = features['KitchenQual'].fillna("TA")

    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    

    # the data description stats that NA refers to "No Pool"

    features["PoolQC"] = features["PoolQC"].fillna("None")

    # Replacing the missing values with 0, since no garage = no cars in garage

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

        features[col] = features[col].fillna(0)

    # Replacing the missing values with None

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

        features[col] = features[col].fillna('None')

    # NaN values for these categorical basement features, means there's no basement

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

        features[col] = features[col].fillna('None')

        

    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood

    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



    # We have no particular intuition around how to fill in the rest of the categorical features

    # So we replace their missing values with None

    objects = []

    for i in features.columns:

        if features[i].dtype == object:

            objects.append(i)

    features.update(features[objects].fillna('None'))

        

    # And we do the same thing for numerical features, but this time with 0s

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    numeric = []

    for i in features.columns:

        if features[i].dtype in numeric_dtypes:

            numeric.append(i)

    features.update(features[numeric].fillna(0))    

    return features



all_features = handle_missing(all_features)
# Let's make sure we handled all the missing values

missing = percent_missing(all_features)

df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)

print('Percent of missing data')

df_miss[0:10]
# Fix skewed features

# Fetch all numeric features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric = []

for i in all_features.columns:

    if all_features[i].dtype in numeric_dtypes:

        numeric.append(i)
# Find skewed numerical features

skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))

skewness = pd.DataFrame({'Skew' :high_skew})

skew_features.head(10)
# Normalize skewed features

for i in skew_index:

    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
all_features['BsmtFinType1_Unf'] = 1*(all_features['BsmtFinType1'] == 'Unf')

all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0) * 1

all_features['HasOpenPorch'] = (all_features['OpenPorchSF'] == 0) * 1

all_features['HasEnclosedPorch'] = (all_features['EnclosedPorch'] == 0) * 1

all_features['Has3SsnPorch'] = (all_features['3SsnPorch'] == 0) * 1

all_features['HasScreenPorch'] = (all_features['ScreenPorch'] == 0) * 1

all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)

all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']

all_features = all_features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']

all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']



all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +

                                 all_features['1stFlrSF'] + all_features['2ndFlrSF'])

all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +

                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))

all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +

                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +

                              all_features['WoodDeckSF'])

all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)

all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)

all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)

all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)



all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_features['has2ndfloor'] = all_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
def logs(res, ls):

    m = res.shape[1]

    for l in ls:

        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   

        res.columns.values[m] = l + '_log'

        m += 1

    return res



log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']



all_features = logs(all_features, log_features)
def squares(res, ls):

    m = res.shape[1]

    for l in ls:

        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   

        res.columns.values[m] = l + '_sq'

        m += 1

    return res 



squared_features = ['YearRemodAdd', 'LotFrontage_log', 

              'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',

              'GarageCars_log', 'GarageArea_log']

all_features = squares(all_features, squared_features)
all_features = pd.get_dummies(all_features).reset_index(drop=True)

all_features.shape
all_features.head()
all_features.shape
# Remove any duplicated column names

all_features = all_features.loc[:,~all_features.columns.duplicated()]
X = all_features.iloc[:len(train_labels), :]

X_test = all_features.iloc[len(train_labels):, :]

X.shape, train_labels.shape, X_test.shape



X_train, X_valid, y_train, y_valid = train_test_split(X, train_labels, test_size=0.15, random_state=42)
# Please turn the 'Internet' toggle On in the Settings panel to your left, in order to make changes to this kernel.

!pip install wandb -q
# WandB

import wandb

import keras

from wandb.keras import WandbCallback

from sklearn.model_selection import cross_val_score

# Import models (add your models here)

from sklearn import svm

from sklearn.linear_model import Ridge, RidgeCV

from xgboost import XGBRegressor
# Initialize wandb run

wandb.init(anonymous='allow', project="pick-a-model")
%%wandb

# Initialize and fit model (add your classifier here)

svr = svm.SVR(C= 20, epsilon= 0.008, gamma=0.0003)

svr.fit(X_train, y_train)



# Get CV scores

scores = cross_val_score(svr, X_train, y_train, cv=5)



# Log scores

for score in scores:

    wandb.log({'cross_val_score': score})
# Initialize wandb run

wandb.init(anonymous='allow', project="pick-a-model")
%%wandb

# Initialize and fit model (add your classifier here)

xgb = XGBRegressor(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       random_state=42)

xgb.fit(X_train, y_train)



# Get CV scores

scores = cross_val_score(xgb, X_train, y_train, cv=3)



# Log scores

for score in scores:

    wandb.log({'cross_val_score': score})
# Initialize wandb run

wandb.init(anonymous='allow', project="pick-a-model")
%%wandb

# Initialize and fit model (add your classifier here)

ridge = Ridge(alpha=1e-3)

ridge.fit(X_train, y_train)



# Get CV scores

scores = cross_val_score(ridge, X_train, y_train, cv=5)



# Log scores

for score in scores:

    wandb.log({'cross_val_score': score})
wandb.init(anonymous='allow', project="picking-a-model", name="neural_network")
%%wandb

# Model

model = Sequential()

model.add(Dense(50, input_dim=378, kernel_initializer='normal', activation='relu'))

model.add(Dense(20, kernel_initializer='normal', activation='relu'))

model.add(Dense(1, kernel_initializer='normal'))



# Compile model

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())



model.fit(X_train, y_train, epochs=20, batch_size=10, verbose=0,

        callbacks=[WandbCallback(validation_data=(X_valid, y_valid))])