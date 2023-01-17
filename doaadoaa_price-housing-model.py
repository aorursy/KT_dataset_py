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
df = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

print("Done!")
# Specific X , y

X = df.drop(columns= 'Price' , axis= 1)

y = df.Price
# Now I split the Data

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Create Model

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# function for comparing different approaches

def score_dataset(train_X, val_X, train_y, val_y):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(train_X, train_y)

    preds = model.predict(val_X)

    return mean_absolute_error(val_y, preds)
# Drop missing Data

cols_with_missing = [col for col in df.columns

                     if df[col].isnull().any()]

reduced_X_train = train_X.drop(cols_with_missing, axis=1)

reduced_X_valid = val_X.drop(cols_with_missing, axis=1)
missing_value = [col for col in train_X if train_X[col].isnull().any()]

if missing_value == [] :

    print("Not Exist Missing Value!")

print("Done!")
# Specific object Columns To Start Encoding

s = (reduced_X_train.dtypes == 'object')

object_cols = list(s[s].index)
# Start Encoding For Category Columns

from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(reduced_X_train[object_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(reduced_X_valid[object_cols]))

OH_cols_train.index = reduced_X_train.index

OH_cols_valid.index = reduced_X_valid.index

num_train_X = reduced_X_train.drop(object_cols, axis=1)

num_val_X = reduced_X_valid.drop(object_cols, axis=1)

OH_train_X = pd.concat([num_train_X, OH_cols_train], axis=1)

OH_val_X = pd.concat([num_val_X, OH_cols_valid], axis=1)
res = score_dataset(OH_train_X, OH_val_X, train_y, val_y)

print(res)