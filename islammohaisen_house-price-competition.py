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
# load data

train_df = pd.read_csv('../input/homeprice/train.csv')

test_df = pd.read_csv('../input/homeprice/test.csv')
train_df.shape, test_df.shape
train_df.info()

test_df.info()
# Missing values imputer in train values

list_obj_col = list(train_df.select_dtypes(include = 'object').columns)

list_num_col = list(train_df.select_dtypes(exclude= 'object').columns)

def fillna_all(df):

  for col in list_obj_col:

      df[col].fillna(value= df[col].mode()[0],inplace = True)

  for col in list_num_col:

    df[col].fillna(value=df[col].mean(), inplace=True)

    

fillna_all(train_df)
train_df.info()  # now, there are no missing values
for col in list_obj_col:

    test_df[col].fillna(value= train_df[col].mode()[0],inplace = True)

list_num_colt = list(test_df.select_dtypes(exclude= 'object').columns)

for col in list_num_colt:

    test_df[col].fillna(value= train_df[col].mean(),inplace = True)
test_df.info()    # now, there are no missing values
# unique values

for col in list_obj_col:

    uni_train = col,':', train_df[col].nunique()

    uni_test= col,':', test_df[col].nunique()

    if uni_train != uni_test:

       print (uni_train,uni_test, True)

# finding not equal unique values in the test and train values in the same column and drop that column
train_df.drop(['Utilities', 'Condition2', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating', 'Electrical', 'GarageQual', 'PoolQC','MiscFeature'], axis=1, inplace =True)
test_df.drop(['Utilities', 'Condition2', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating', 'Electrical', 'GarageQual', 'PoolQC','MiscFeature'], axis=1, inplace =True)
list_obj_col1 = list(train_df.select_dtypes(include = 'object').columns)
train_df.shape, test_df.shape   # after dropping some catagorical columns
# OneHotencoder: enconding the remaining catagorical columns
dummy = pd.get_dummies(train_df[list_obj_col1], prefix = list_obj_col1)

train_df.drop(list_obj_col1, axis=1, inplace=True)

train_df_final= pd.concat([train_df,dummy], axis =1)
dummy1 = pd.get_dummies(test_df[list_obj_col1], prefix = list_obj_col1)

test_df.drop(list_obj_col1, axis=1, inplace=True)

test_df_final= pd.concat([test_df,dummy1], axis =1)
train_df_final.shape, test_df_final.shape
y = train_df_final['SalePrice']

x = train_df_final.iloc[:,:-1]
x.shape, y.shape
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.2, random_state=0)
train_x.shape, test_x.shape, train_y.shape, test_y.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor(n_estimators=100, random_state=23)

model.fit(train_x, train_y)

preds = model.predict(test_x)

print(mean_absolute_error(preds, test_y))
from xgboost import XGBRegressor

regressor1 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

regressor1.fit(train_x, train_y)

predictions1 = regressor1.predict(test_x)

print(mean_absolute_error(predictions1, test_y))
output = pd.DataFrame({'Id': test_x.index,

                       'SalePrice': predictions1})

output.to_csv('submission.csv', index=False)