# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
house = pd.read_csv("/kaggle/input/streeteasy/streeteasy.csv")

house.head()
house.isnull().any()
house.dtypes
rent_avg = house.groupby("neighborhood")['rent'].mean().reset_index()

rent_avg
sns.set(style="whitegrid")

ax = sns.boxplot(y="rent", data=house)
rent_avg_wealthy10 = rent_avg.sort_values(by=['rent']).tail(10)

sns.set(style="whitegrid")

ax = sns.barplot(x="rent", y="neighborhood", data=rent_avg_wealthy10, errwidth=30)
rent_avg_working10 = rent_avg.sort_values(by=['rent']).head(10)

sns.set(style="whitegrid")

ax = sns.barplot(x="rent", y="neighborhood", data=rent_avg_working10, errwidth=30)
bedrooms = house.groupby("bedrooms")['rent'].mean().reset_index()

bedrooms_barchart = bedrooms.sort_values(by=['rent'])

sns.set(style="whitegrid")

ax = sns.barplot(x="bedrooms", y="rent", data=bedrooms_barchart, errwidth=30)
bathrooms = house.groupby("bathrooms")['rent'].mean().reset_index()

bathrooms_barchart = bathrooms.sort_values(by=['rent'])

sns.set(style="whitegrid")

ax = sns.barplot(x="bathrooms", y="rent", data=bathrooms_barchart, errwidth=30)
floor = house.groupby("floor")['rent'].mean().reset_index()

floor_barchart = floor.sort_values(by=['rent'])

sns.set(style="whitegrid")

ax = sns.barplot(x="floor", y="rent", data=floor_barchart, errwidth=30)
min_to_subway = house.groupby("min_to_subway")['rent'].mean().reset_index()

min_to_subway_barchart = min_to_subway.sort_values(by=['rent'])

sns.set(style="whitegrid")

ax = sns.barplot(x="min_to_subway", y="rent", data=min_to_subway_barchart, errwidth=30)
ax = sns.scatterplot(x="size_sqft", y="rent", data=house)
ax = sns.scatterplot(x="building_age_yrs", y="rent", data=house)
house.head()
import statsmodels.api as sm

x_var = house.iloc[:, 9:17]

y_var = house[["rent"]]
#posative coef = if x increases, the rent increases

#negative coef = if x increases, the rent decreases

#no fee means no broker's fee is paid

lm = sm.OLS(y_var, x_var, data = house).fit()

print(lm.summary())
print(x_var.shape)

print(y_var.shape)

X=x_var.values

y=y_var.values
print(X.shape)

print(y.shape)
from sklearn import neighbors, preprocessing

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import Normalizer

from sklearn import neighbors

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

scaler = Normalizer().fit(X_train)

house_model = DecisionTreeClassifier()

house_model.fit(X_train, y_train)

house_preds = house_model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, house_preds))