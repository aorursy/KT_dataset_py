# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sea

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df
df.dtypes
# Unnamed: 0 didn't exists in dataframe
df.drop(["id"], axis = 1) 

df.describe()
df["floors"].value_counts().to_frame()
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(df["waterfront"], df["price"])
sns.regplot(df["sqft_above"], df["price"])
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = df[["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]]
Y = df["price"]

lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline

features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(df[features],df['price'])

pipe.score(df[features],df['price'])
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

from sklearn.linear_model import Ridge

RidgeModel = Ridge(alpha=0.1) 
RidgeModel.fit(x_train, y_train)
RidgeModel.score(x_test, y_test)
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[features])
x_test_pr=pr.fit_transform(x_test[features])

RidgeModel = Ridge(alpha=0.1) 
RidgeModel.fit(x_train_pr, y_train)
RidgeModel.score(x_test_pr, y_test)