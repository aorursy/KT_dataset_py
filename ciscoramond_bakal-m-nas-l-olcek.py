# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
X_full = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", index_col="Id")

X_test_full = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", index_col="Id")



X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



no_need = ["Alley","PoolQC","Fence","MiscFeature"]



X_full.drop(no_need, axis=1, inplace=True)

X_test_full.drop(no_need, axis=1, inplace=True)



X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)
categorical_cols = [cname for cname in X_train_full.columns if 

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



numerical_cols = [cname for cname in X_train_full.columns if

                 X_train_full[cname].dtype in ["int64","float64"]]



my_cols = categorical_cols + numerical_cols
X_train = X_train_full.copy()

X_valid = X_valid_full.copy()

X_test = X_test_full.copy()
numerical_transformer = SimpleImputer(strategy="constant")



categorical_transformer = Pipeline(steps=[

    ("imputer", SimpleImputer(strategy="constant")),

    ("onehot", OneHotEncoder(handle_unknown="ignore"))

])
preprocessor = ColumnTransformer(transformers=[

    ("num", numerical_transformer, numerical_cols),

    ("cat", categorical_transformer, categorical_cols)

])
model = XGBRegressor(n_estimators=700,learning_rate=0.05)



clf = Pipeline(steps=[

    ("preprocessor", preprocessor),

    ("model", model)

])



clf.fit(X_train,y_train)



preds = clf.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds})

output.to_csv('submission.csv', index=False)