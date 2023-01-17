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
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df
df.columns
Data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

Data_test
cols_with_missing = [[col,df[col].unique(),df[col].isna().sum()] for col in df.columns

                     if df[col].isnull().any()]

cols_with_missing
drop_col = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
new_df = df.drop(drop_col, axis=1)

new_df
imputed_col = ['LotFrontage','MasVnrArea','GarageYrBlt']
impu_df = df[imputed_col]

impu_df
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_df = pd.DataFrame(my_imputer.fit_transform(impu_df))

# Imputation removed column names; put them back

imputed_df.columns = impu_df.columns
imputed_df
new_df.update(imputed_df)
new_df[imputed_col].isnull().sum()
new_df.isnull().sum()
# Get list of categorical variables

s = (new_df.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
from sklearn.preprocessing import LabelEncoder

cols = object_cols

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(new_df[c].values)) 

    new_df[c] = lbl.transform(list(new_df[c].values))
y = new_df['SalePrice']

X = new_df.drop(columns=['SalePrice'])

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2)
para = list(range(100, 1001, 100))

print(para)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

results = {}

for n in para:

    print('para=', n)

    model = RandomForestRegressor(n_estimators=n, random_state=1)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    mae = mean_absolute_error(y_true=y_valid, y_pred=preds)

    print (mae)

    results[n] = mae

    print('--------------------------')
import matplotlib.pylab as plt

# sorted by key, return a list of tuples

lists = sorted(results.items()) 

p, a = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(p, a)

plt.show()
best_para = min(results, key=results.get)

print('best para', best_para)

print('value', results[best_para])
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



results_XG = {}

for n in para:

    print('para=', n)

    model_XG = XGBRegressor(n_estimators=n,learning_rate=0.05, random_state=1)

    model_XG.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

    preds = model_XG.predict(X_valid)

    mae = mean_absolute_error(y_true=y_valid, y_pred=preds)

    print (mae)

    results_XG[n] = mae

    print('--------------------------')
best_para_XG = min(results_XG, key=results_XG.get)

print('best para', best_para_XG)

print('value', results_XG[best_para_XG])
import matplotlib.pylab as plt

# sorted by key, return a list of tuples

lists = sorted(results_XG.items()) 

p, a = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(p, a)

plt.show()


final_model = XGBRegressor(n_estimators=200,learning_rate=0.05, random_state=1)

model_XG.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

final_preds = model_XG.predict(X_valid)

mae = mean_absolute_error(y_true=y_valid, y_pred=final_preds)

print (mae)

print('--------------------------')
# Save test predictions to file

output = pd.DataFrame({'Id': Data_test.index,

                       'SalePrice': final_preds})

output.to_csv('submission.csv', index=False)