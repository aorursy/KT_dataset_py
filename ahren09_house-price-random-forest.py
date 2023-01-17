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
train_path = "../input/home-data-for-ml-course/train.csv"

test_path = "../input/home-data-for-ml-course/test.csv"

df_train = pd.read_csv(train_path)

df_test = pd.read_csv(test_path)
df_train.columns
from sklearn.model_selection import train_test_split

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = df_train[features]

y = df_train["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

model = RandomForestRegressor(n_estimators=100, random_state=1)

model.fit(X_train, y_train)
y_pred = model.predict(X_val)

error = mean_squared_error(y_pred, y_val)
X_test = df_test[features]

y_test_pred = model.predict(X_test)
y_test_pred
submission = pd.DataFrame({"Id":X_test.index, "SalePrice":y_test_pred})

submission.to_csv("submission.csv", index=False)