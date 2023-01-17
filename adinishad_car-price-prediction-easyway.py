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
train = pd.read_csv("/kaggle/input/car-price-prediction/CarPrice_Assignment.csv")
train.head()
train.describe().T
train.isnull().sum()
train.tail()
train.select_dtypes(object).all()
train.drop(train.select_dtypes(object), axis=1, inplace=True)
train.columns.values
train.drop('car_ID', axis=1, inplace=True)
train.head()
train.shape
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.pairplot(train, x_vars=["wheelbase", "carlength", "horsepower"], y_vars="price", size=7, aspect=0.7, kind="reg")
corr_mat = train.corr()
corr_mat["price"].sort_values(ascending=False)
fig = plt.subplots(nrows=1, figsize=(12,8))
sns.heatmap(train.corr(), cmap="YlGnBu", annot=True, linewidths=2, linecolor='black')
from sklearn.model_selection import train_test_split
X = train.drop('price', axis=1)
y = train["price"].copy()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
predict = model.predict(X_test)
predict
list(y_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predict)
mse
rmse = np.sqrt(mse)
rmse
r2_score(y_test, predict)