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
from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
train_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')

test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')
y = train_data.SalePrice

X_full = train_data.copy().drop('SalePrice', axis=1)

X_test = test_data.copy()
full_num_cols = [col for col in X_full.columns if X_full[col].dtype in ['int64', 'float64']]

full_cat_cols = [col for col in X_full.columns if X_full[col].dtype == 'object']



low_cardi_cats = [col for col in full_cat_cols if X_full[col].nunique() < 10]

high_cardi_cats = [col for col in full_cat_cols if X_full[col].nunique() > 10]



nums_with_missing = [col for col in full_num_cols if X_full[col].isnull().any()]

cats_with_missing = [col for col in full_cat_cols if X_full[col].isnull().any()]



count_num_missing = {col: X_full[col].isnull().sum() for col in nums_with_missing}

count_cat_missing = {col: X_full[col].isnull().sum() for col in cats_with_missing}
high_columns = {c: X_full[c].nunique() for c in high_cardi_cats}

print(high_columns)
X = X_full.copy()
encoder = LabelEncoder()

for c in high_cardi_cats:

    X[c] = encoder.fit_transform(X[c])
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

cat_cols = [col for col in X.columns if X[col].dtype == 'object']
num_transformer = SimpleImputer(strategy='mean')



low_cat_transformer = make_pipeline(SimpleImputer(strategy='constant'),

                                OneHotEncoder(handle_unknown='ignore'))



high_cat_transformer = make_pipeline(SimpleImputer(strategy='constant'),

                                     OrdinalEncoder(dtype=pd.DataFrame))



preprocessing = ColumnTransformer(

    transformers=[

    ('num', num_transformer, num_cols),

    ('low_cat', low_cat_transformer, low_cardi_cats)

])
#Apply new parameters to the pipeline

model = XGBRegressor(n_estimators=300, max_depth=2)



pipeline = Pipeline(steps=[

    ('model', model)

])
imputer = SimpleImputer(strategy='constant')

X_test[high_cardi_cats] = pd.DataFrame(imputer.fit_transform(X_test[high_cardi_cats]))



for c in high_cardi_cats:

    X_test[c] = encoder.fit_transform(X_test[c])
X_final = pd.DataFrame(preprocessing.fit_transform(X))

X_test_final = pd.DataFrame(preprocessing.transform(X_test))
#Fit

pipeline.fit(X_final,y)
preds = pipeline.predict(X_test_final)
#Save predictions to submission

output = pd.DataFrame({'Id' : X_test.index, 'SalePrice' : preds})

output.to_csv('submission', index=False)