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
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
#Read the data

train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv',index_col='Id')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
# Select categorical columns with relatively low cardinality (low number of unique values)

categorical_cols = [cname for cname in train.columns if

                    train[cname].nunique() < 10 and 

                    train[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in train.columns if 

                train[cname].dtype in ['int64', 'float64']]
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, RobustScaler

from sklearn.metrics import mean_absolute_error

from sklearn.compose import ColumnTransformer



# Create a pipeline for numerical_columns

numerical_transformer = Pipeline(steps=[

    ('num_imputer', SimpleImputer(strategy='median')),

    ('num_scaler', RobustScaler())

])



# Create a pipeline for categorical_columns

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

])



# Apply these pipelines to the columns

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
import seaborn as sns

fig, (axis1) = plt.subplots(1,1,figsize=(20,10))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
c_feat = list(train.select_dtypes(['float64', 'int64']).columns)

c_data = train[c_feat].fillna(0)
y = c_data.SalePrice

X = c_data[c_feat[:-1]]

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, random_state=1, test_size=0.3)
re_or = GradientBoostingRegressor(random_state=1)

re_or.fit(X_tr, y_tr)

y_pred = re_or.predict(X_ts)

er_mae = mean_absolute_error(y_pred, y_ts)

print(er_mae)
model_on_full_data = GradientBoostingRegressor(random_state=1)

model_on_full_data.fit(X, y)



Xt = test[c_feat[:-1]]

Xt = Xt.fillna(0)



test_preds = model_on_full_data.predict(Xt)

output = pd.DataFrame({'Id': test.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)