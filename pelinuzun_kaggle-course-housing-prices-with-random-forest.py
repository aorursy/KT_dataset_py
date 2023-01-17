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
train_data = pd.read_csv("../input/home-data-for-ml-course/train.csv")
train_data.head()
target_data = train_data["SalePrice"]

target_data.head()
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = train_data[features]

X.head()
X.info()
from sklearn.model_selection import train_test_split



X_train, X_test, x_val, y_val = train_test_split(X, target_data, random_state=1)

from sklearn.ensemble import RandomForestRegressor



rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(X_train, x_val)
prediction = rf_model.predict(X_test)
prediction[0:5]
from sklearn.metrics import mean_absolute_error



mean_error = mean_absolute_error(prediction, y_val)

mean_error
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': prediction})

output.to_csv('submission.csv', index=False)