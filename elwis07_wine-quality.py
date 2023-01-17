# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/winequalityN.csv")
data.info()

data = data.dropna()
print("data shape :", data.shape)
categoricals = data.select_dtypes(include='object').columns

numericals = data.select_dtypes(exclude='object').columns

print(f'{len(categoricals)} categorical features')

print(f'{len(numericals)} numerical features')
import matplotlib.pyplot as plt

data[(data.dtypes[(data.dtypes == "float")|

                              (data.dtypes == "int64")].index.values)].hist(figsize=[14,14])
y = data.quality

X = data.drop("quality", axis = 1)
X = pd.get_dummies(X)

X = X.drop("type_red", axis = 1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y= train_test_split(X, y, test_size = 0.25)
from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=1000)

my_model.fit(train_X, train_y, 

             early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], 

             verbose=False)
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(test_X)

predictions = np.rint(predictions)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, test_y)))
result = pd.DataFrame({ "predictions" : predictions})
result