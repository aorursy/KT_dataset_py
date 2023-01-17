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
train_data_path = '../input/house-prices-data/train.csv'

test_data_path = '../input/house-prices-data/test.csv'

train = pd.read_csv(train_data_path)

test = pd.read_csv(test_data_path)

# for making train and test equal delete the last row:

train = train.drop([1459])
train.Utilities.value_counts()
train.Alley.value_counts()
train.Street.value_counts()
train.LotFrontage.value_counts()
sum(train.LotFrontage.isna())
train.Neighborhood.value_counts()
col_del = ['Utilities','Alley','Street','PavedDrive','SaleType',

           'Condition1','Condition2','RoofMatl','MasVnrType',

           'BsmtFinSF1','BsmtFinSF2','Heating','CentralAir',

           'Electrical','LowQualFinSF','BsmtHalfBath','KitchenAbvGr',

           'Functional','GarageQual','GarageCond','EnclosedPorch',

           'ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal']



# Deleting bad distribution columns

train = train.drop(columns = col_del)

test = test.drop(columns = col_del)

import category_encoders as ce





cat_features = ['MSZoning','ExterQual','LotShape','LandContour','LotConfig',

                   'LandSlope','BldgType','HouseStyle','RoofStyle',

                    'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure',

                    'BsmtFinType1','BsmtFinType2','HeatingQC',

                    'KitchenQual','FireplaceQu','GarageType','GarageFinish', 

                    'SaleCondition','Exterior1st','Exterior2nd']





# Create the encoder



target_enc = ce.TargetEncoder(cols=cat_features)

target_enc.fit(train[cat_features], train['SalePrice'])

# Transform the features,and replace with Neighborhood in dataframe

train_TE = target_enc.transform(train[cat_features])

test_TE =  target_enc.transform(test[cat_features])

                                

train_TE = train_TE.fillna(0)

test_TE = test_TE.fillna(0)
target=train.SalePrice

train=train.drop(columns = ['SalePrice'])



target
train_TE.to_csv('./train_v2.csv')

test_TE.to_csv('./test_v2.csv')
from sklearn.linear_model import LinearRegression

# MSE is a modlue to calculate how good our model is:

from sklearn.metrics import mean_squared_error as MSE
model = LinearRegression(fit_intercept=True,normalize=False)

model.fit(train_TE,target)

result = model.predict(test_TE)

sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sample.SalePrice = result

sample.to_csv('./sample_1',index=False)

error = MSE(result,target)

error # :||||||