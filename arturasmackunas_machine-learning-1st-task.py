# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas

df_train = pandas.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df_test = pandas.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
pandas.set_option('display.max_rows', 500)

pandas.set_option('display.max_columns',500)

pandas.set_option('display.width',1000)
df_train.shape
df_test.shape
def extract_features(df):

    """

    Input: df, input data frame

    Output: df_features, features for machine learning

    """

    numerical_columns=[

    column_name

    for column_name in df.columns

    if df.dtypes[column_name] in (np.int64, np.float64)

    if column_name != 'SalePrice'

] # list comprehension

    

    df_features = df.loc[:,numerical_columns].fillna(0)

    

    df_features['MSZoning'] = 0 

    for value_index, value in enumerate(('RL', 'RM', 'FV', 'RH', 'C (all)')):

        df_features.loc[df['MSZoning'] == value, 'MSZoning'] = value_index

        

    # todo: add new features

    

    return df_features
X_train = extract_features(df_train)

X_test = extract_features(df_test)



y_train = df_train['SalePrice']
X_train
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor()

model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

df_submission = pandas.DataFrame({'Id': X_test['Id'], 'SalePrice': y_predicted})

df_submission.to_csv('submission.csv', index = False)
!head submission.csv
df_train['MSZoning'].value_counts()
df_train.dtypes
df_train.T