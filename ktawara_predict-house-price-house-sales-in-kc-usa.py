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
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

df = pd.read_csv("../input/kc_house_data.csv")
display(df.head())
display(df.tail())
print(df.info())
print("Data shape: {}" .format(df.shape))
df.describe()
#Take a look at the heat map with seaborn
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#Another way to look at the heat map 
df.corr().style.background_gradient().format('{:.2f}')
#Check price and the scatter plot of each feature
df2 = df.drop(['zipcode','id'],axis=1)

for c in df2.columns:
    if (c != 'price') & (c != 'date'):
        df2[[c,'price']].plot(kind='scatter',x=c,y='price')
df.date.head()
#date conversion
pd.to_datetime(df.date).head()
#df_en_fin = df.drop(['date','zipcode','sqft_living15','sqft_lot15'],axis=1)
#1.Linear regression

df = pd.read_csv("../input/kc_house_data.csv")
X = df.drop(["id", "price", "zipcode", "date"], axis=1)
y = df["price"]

regr = LinearRegression()
scores = cross_val_score(regr, X, y, cv=10)
print("score: %s"%scores.mean())
#2.Random Forest

df = pd.read_csv("../input/kc_house_data.csv")
X = df.drop(["id", "price", "zipcode", "date"], axis=1)
y = df["price"]

regr = RandomForestRegressor()
scores = cross_val_score(regr, X, y, cv=10)
print("score: %s"%scores.mean())
#3.gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("../input/kc_house_data.csv")
X = df.drop(["id", "price", "zipcode", "date"], axis=1)
y = df["price"]

gbrt = GradientBoostingClassifier()
scores = cross_val_score(regr, X, y, cv=10)
print("score: %s"%scores.mean())
#4.k neighborhood method
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("../input/kc_house_data.csv")
X = df.drop(["id", "price", "zipcode", "date"], axis=1)
y = df["price"]

n_neighbors = KNeighborsClassifier()
scores = cross_val_score(regr, X, y, cv=10)
print("score: %s"%scores.mean())
