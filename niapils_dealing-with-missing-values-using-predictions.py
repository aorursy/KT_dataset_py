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
# Import Packages

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor



# Load Data

data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

X_full = data.drop('Price', axis=1)



# use only numeric features for simplicity

num_features = [col for col in X_full.columns if X_full[col].dtypes in ['float64', 'int64']]



# Separate target variable and features

y = data.Price

X = data[num_features]





# Split data into training and validation sets

train_X, valid_X, train_y, valid_y = train_test_split(X, y, train_size=0.7, random_state=1)
def get_score(train_X, train_y, valid_X, valid_y):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(train_X, train_y)

    preds = model.predict(valid_X)

    score = mean_absolute_error(valid_y, preds)

    return score





def predict_imputation(data, variable, features):

    '''Variable is the varaible for missing values are to be predicted for its missing values. 

    Features are the columns that will be used as features in predicting values for the missing values

    '''

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    missing = data[data[variable].isnull()]

    not_missing = data[data[variable].notnull()]

    target = not_missing[variable]

    model.fit(not_missing[features], target)

    pred = pd.DataFrame({variable:model.predict(missing[features])}, index=missing.index)

    original = pd.DataFrame({variable:target}, index=not_missing.index)

    full = pd.concat([pred, original]).sort_index()

    return full
## Get columns with missing values

X.isnull().sum().sort_values(ascending=False).head()

cols_with_missing = ['BuildingArea', 'YearBuilt', 'Car']





## Use the most correlated features with the target variaable for the predictions.

corr = data.corr().round(2)

print(corr)
# METHOD 1: PREDICTION IMPUTATION



# Building Area

variable = 'BuildingArea'

features = ['Landsize', 'Rooms', 'Bathroom', 'Distance']

building_area = predict_imputation(data=data, variable=variable, features=features)



## YearBuilt

variable = 'YearBuilt'

features = ['Price', 'Bathroom', 'Distance']

yearBuilt = predict_imputation(data=data, variable=variable, features=features)



## Car

variable = 'Car'

features = ['Price', 'Bathroom', 'Distance', 'Rooms']

car = predict_imputation(data=data, variable=variable, features=features)



# Put them together

pred_data = pd.concat([X.drop(cols_with_missing, axis=1), building_area, yearBuilt, car], axis=1)



# Split into training and validation sets 

train_X_pred, valid_X_pred, train_y1, valid_y1 = train_test_split(pred_data, y, train_size=0.7, random_state=1)

### METHOD 2: DROPPING COLUMNS WITH MISSING VALUES

train_X_drop, valid_X_drop = train_X.dropna(axis=1), valid_X.dropna(axis=1)





### METHOD 3: SIMPLE IMPUTATION

imputer = SimpleImputer()

train_X_simple = pd.DataFrame(imputer.fit_transform(train_X))

valid_X_simple = pd.DataFrame(imputer.transform(valid_X))



# Fix index and columns

train_X_simple.columns, train_X_simple.index = num_features, train_X.index

valid_X_simple.columns, valid_X_simple.index = num_features, valid_X.index







### METHOD 4: EXTENDED IMPUTATION

imputer = SimpleImputer(add_indicator=True)

train_X_extend = pd.DataFrame(imputer.fit_transform(train_X))

valid_X_extend = pd.DataFrame(imputer.transform(valid_X))



# Fix index and columns

train_X_extend.columns, train_X_extend.index = num_features + ['ind1', 'ind2', 'ind3'], train_X.index

valid_X_extend.columns, valid_X_extend.index = num_features + ['ind1', 'ind2', 'ind3'], valid_X.index
### Compare performance of the all methods

score = {}

score['drop_score'] = get_score(train_X_drop, train_y, valid_X_drop, valid_y)

score['simple_score'] = get_score(train_X_simple, train_y, valid_X_simple, valid_y)

score['extend_score'] = get_score(train_X_extend, train_y, valid_X_extend, valid_y)

score['pred_score'] = get_score(train_X_pred, train_y1, valid_X_pred, valid_y)



print(score)