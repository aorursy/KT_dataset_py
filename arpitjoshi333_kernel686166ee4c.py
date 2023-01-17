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
X_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
y = y.values
X_full.drop(['SalePrice'], axis=1, inplace=True)
X_test_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')
list1 = ['GarageFinish', 'PoolQC', 'BsmtFinType2', 'Fence', 
         'MasVnrType', 'BsmtFinType1', 'BsmtCond', 'MiscFeature',
         'GarageType', 'Alley', 'GarageCond', 'BsmtQual', 
         'GarageQual', 'FireplaceQu', 'BsmtExposure' ]

le_list  = ['GarageFinish', 'PoolQC', 'BsmtFinType2', 'BsmtFinType1',
            'BsmtCond', 'Alley', 'GarageCond', 'BsmtQual', 'RoofStyle',
            'GarageQual', 'FireplaceQu', 'BsmtExposure', 'HeatingQC',
            'Foundation', 'SaleCondition', 'LotConfig', 'Heating',
            'ExterQual', 'LandSlope', 'PavedDrive', 'LandContour', 
            'Utilities', 'Functional', 'CentralAir', 'Exterior1st',
            'LotShape', 'RoofMatl', 'Exterior2nd', 'KitchenQual',
            'ExterCond', 'Street']

one_list = ['Fence', 'MasVnrType', 'MiscFeature', 'GarageType', 'Electrical',
            'Condition1', 'Condition2', 'SaleType', 'BldgType', 'HouseStyle',
            'MSZoning', 'Neighborhood']
for col in list1:
    X_full[col].fillna(value='none', inplace=True)
    X_test_full[col].fillna(value='none', inplace=True)
from sklearn.impute import SimpleImputer as IMP
imp = IMP(strategy="most_frequent")
X = pd.DataFrame(imp.fit_transform(X_full))
X.columns = X_full.columns
X_test = pd.DataFrame(imp.transform(X_test_full))
X_test.columns = X_test_full.columns
from sklearn.preprocessing import LabelEncoder
for col in le_list:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.compose import ColumnTransformer as ColT
ct = ColT([("Encoding", OHE(), one_list)], remainder='passthrough' )
X = ct.fit_transform(X)
X_test = ct.transform(X_test)
X = np.array(X, dtype=np.float)
X_test = np.array(X_test, dtype=np.float)
from sklearn.model_selection import train_test_split as tts
X_train, X_val, y_train, y_val = tts(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_val = sc_X.transform(X_val)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_val = sc_y.transform(y_val.reshape(-1,1))
y_train = np.ravel(y_train)
y_val = np.ravel(y_val)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
score_dataset(X_train, X_val, y_train, y_val)
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_test)
predictions = sc_y.inverse_transform(preds)
submission = pd.read_csv("/kaggle/input/home-data-for-ml-course/sample_submission.csv")

submission.iloc[:, 1] = np.floor(predictions)

submission.to_csv("House_price_submission_v17.csv", index=False)
