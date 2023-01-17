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

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV

import xgboost as xgb



%matplotlib inline
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_data['train'] = 1

test_data['train'] = 0
#concatenate the data

full_data = pd.concat([train_data, test_data], sort = False, ignore_index = True)
full_data['SalePrice'].replace(np.nan, 1, inplace = True)
total = full_data.isnull().sum().sort_values(ascending = False)

percent = (full_data.isnull().sum()/full_data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

missing_data.head(20)

#It's not going to affect the model if we remove the columns which has more than 80% missing values.

full_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis = 1, inplace = True)



# As NA in means NoFireplace in FirePlaceQU 

full_data['FireplaceQu'].fillna('No', inplace = True)



#fixing the LotFrontage column

full_data["LotFrontage"] = full_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



#In garage related columns NA means No Garage

for col in ('GarageFinish', 'GarageQual', 'GarageCond', 'GarageType'):

    full_data[col].fillna('No', inplace = True)



#GarageYrBuilt

full_data['GarageYrBlt'].fillna(0, inplace = True)



#In basement related columns NA means No basement

for col in ('BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2'):

    full_data[col].fillna('No', inplace = True)



#MasVnrType and MasVnrArea columns

full_data['MasVnrType'].fillna('No', inplace = True)

full_data['MasVnrArea'].fillna(0, inplace = True)



#In MSZoning column , replace the null values with RL as it has the highest frequency.

full_data['MSZoning'].fillna('RL', inplace = True)



#BsmtFullBath and BsmtHalfBath columns

full_data['BsmtFullBath'].fillna(0, inplace = True)

full_data['BsmtHalfBath'].fillna(0, inplace = True)



#As AllPub has a very high frequnecy in Utilities its good to remove this column.

full_data.drop(['Utilities'], axis = 1, inplace = True)



#In functional column we can replace Null values with Typ as it has a very high frequncy.

full_data['Functional'].fillna('Typ', inplace = True)



#GarageArea column 

full_data['GarageArea'].fillna(0, inplace = True)



full_data['TotalBsmtSF'].fillna(0, inplace = True)



#Replace NA with Sbrkr as it has the highest frequency. 

full_data['Electrical'].fillna('SBrkr', inplace = True)

full_data['Electrical'].replace(['Mix', 'FuseP','FuseF'], ['SBrkr', 'SBrkr', 'SBrkr'], inplace = True)



#Let's drop Exterior1st and Exterior2nd column as i dont think this feature has much importance.

full_data.drop(['Exterior1st', 'Exterior2nd'], axis = 1, inplace = True)

full_data['KitchenQual'].fillna('TA', inplace = True)



#GarageCars column

full_data['GarageCars'].fillna(0, inplace = True)



#I also think BsmtUnfSF is of no importance in predicting column, so i am going to remove it.

full_data.drop(['BsmtUnfSF'], axis = 1, inplace = True)



#We're also going to remove BsmtFinSF1 and BsmtFinSF2 as it's have only zeros like 90% values are zeros only.

full_data.drop(['BsmtFinSF1', 'BsmtFinSF2'], axis = 1, inplace = True)

full_data.drop(['Neighborhood', 'Street', 'Condition2', 'RoofMatl', 'Heating'], axis = 1, inplace = True)



#For column salatype

full_data['SaleType'].fillna('WD', inplace = True)
full_data.skew()
#Apply Log Transformations

full_data['GrLivArea'] = np.log(full_data['GrLivArea'])

full_data['LotArea'] = np.log1p(full_data['LotArea'])

full_data['LowQualFinSF'] = np.log1p(full_data['LowQualFinSF'])

full_data['BsmtHalfBath'] = np.log1p(full_data['BsmtHalfBath'])

full_data['KitchenAbvGr'] = np.log1p(full_data['KitchenAbvGr'])

full_data['GarageYrBlt'] = np.log1p(full_data['GarageYrBlt'])

full_data['EnclosedPorch'] = np.log1p(full_data['EnclosedPorch'])

full_data['3SsnPorch'] = np.log1p(full_data['3SsnPorch'])

full_data['ScreenPorch'] = np.log1p(full_data['ScreenPorch'])

full_data['PoolArea'] = np.log1p(full_data['PoolArea'])

full_data['MiscVal'] = np.log1p(full_data['MiscVal'])
#Replace categorical columns

cat_to_num = {

    'HouseStyle' : {'1Story' : 1, '1.5Unf' : 2, '1.5Fin' : 3, '2Story' : 4, '2.5Unf' : 5,

                    '2.5Fin' : 6, 'SFoyer' : 7, 'SLvl' : 8},

    'ExterQual' : {'Ex' : 10, 'Gd' : 8, 'TA' : 6, 'Fa' : 4, 'Po' : 2},

    'ExterCond' : {'Ex' : 10, 'Gd' : 8, 'TA' : 6, 'Fa' : 4, 'Po' : 2},

    'BsmtQual' : {'Ex' : 10, 'Gd' : 8, 'TA' : 6, 'Fa' : 4, 'Po' : 2, 'No' : 0},

    'BsmtCond' : {'Ex' : 10, 'Gd' : 8, 'TA' : 6, 'Fa' : 4, 'Po' : 2, 'No' : 0},

    'BsmtExposure' : {'Gd' : 10, 'Av' : 7, 'Mn' : 4, 'No' : 0},

    'BsmtFinType1' : {'GLQ' : 10, 'ALQ' : 8, 'BLQ' : 6, 'Rec' : 5, 'LwQ' : 4, 'Unf' : 2, 'No' : 0},

    'BsmtFinType2' : {'GLQ' : 10, 'ALQ' : 8, 'BLQ' : 6, 'Rec' : 5, 'LwQ' : 4, 'Unf' : 2, 'No' : 0},

    'HeatingQC' : {'Ex' : 10, 'Gd' : 8, 'TA' : 6, 'Fa' : 4, 'Po' : 2},

    'KitchenQual' : {'Ex' : 10, 'Gd' : 8, 'TA' : 6, 'Fa' : 4, 'Po' : 2},

    'FireplaceQu' : {'Ex' : 10, 'Gd' : 8, 'TA' : 6, 'Fa' : 4, 'Po' : 2, 'No' : 0},

    'GarageQual' : {'Ex' : 10, 'Gd' : 8, 'TA' : 6, 'Fa' : 4, 'Po' : 2, 'No' : 0},

    'GarageCond' : {'Ex' : 10, 'Gd' : 8, 'TA' : 6, 'Fa' : 4, 'Po' : 2, 'No' : 0}

}

full_data.replace(cat_to_num, inplace = True)
threshold = 50

cat_columns = list(full_data.select_dtypes(include = ['object', 'category']).columns)

for col in cat_columns:

    vc = full_data[col].value_counts()

    val_remove = vc[vc <= threshold].index.values

    full_data[col].loc[full_data[col].isin(val_remove)] = 'Other'
#convert categorical variables into dummies

full_data = pd.get_dummies(full_data)
full_data.drop(['Id'], axis = 1, inplace = True)



df_train = full_data[full_data['train'] == 1]

df_train = df_train.drop(['train'],axis=1)

df_train.drop(['SalePrice'], axis = 1, inplace = True)



df_test = full_data[full_data['train'] == 0]

df_test = df_test.drop(['train'],axis=1)

df_test.drop(['SalePrice'], axis = 1, inplace = True)
#Target variable

Y = full_data['SalePrice'][full_data['train'] == 1]
#split the training data

X_train, X_val, Y_train, Y_val = train_test_split(df_train, Y, test_size = 0.15, random_state = 4)
#Train a model using XGBRegressor

xgb_reg = xgb.XGBRegressor( booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.6, gamma=0,

             importance_type='gain', learning_rate=0.01, max_delta_step=0,

             max_depth=4, min_child_weight=1.5, n_estimators=2400,

             n_jobs=1, nthread=None, objective='reg:linear',

             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 

             silent=None, subsample=0.8, verbosity=1)
xgb_reg.fit(X_train, Y_train, verbose = 1, early_stopping_rounds = 50, eval_metric = 'rmse' , eval_set = [(X_val, Y_val)])
Y_hat1 = xgb_reg.predict(df_test)
submission = pd.DataFrame({

        "Id": test_data["Id"],

        "SalePrice": Y_hat1

    })

submission.to_csv('submission.csv', index=False)