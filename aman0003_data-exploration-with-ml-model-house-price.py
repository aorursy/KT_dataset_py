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
df = pd.read_csv("../input/kc_house_data.csv")
df.head()
df.shape
df.isnull().sum()

unique_df = pd.DataFrame(columns = ["features", "unique_count"])
for col in df.columns:
    unique_df = unique_df.append({"features": col, "unique_count" : len(df[col].unique())}, ignore_index = True)
unique_df
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 12))
corr = df.corr()
sns.heatmap(corr, annot = True,fmt='.2f',
    cmap=sns.diverging_palette(20, 220, n=200),square=True)
print(pd.Series(df["bathrooms"]).value_counts())
print(pd.Series(df["floors"]).value_counts())
print(pd.Series(df["waterfront"]).value_counts())
print(pd.Series(df["view"]).value_counts())
print(pd.Series(df["grade"]).value_counts())
from sklearn import preprocessing
x = df[["grade"]].values

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)
df["grade"] = df_normalized
df["any_view"] = df["view"].apply(lambda x : 1 if x > 0 else 0)
df["no_of_views"] = df["view"]
df.drop(["view"], axis = 1, inplace = True)
df.drop(df[df["bedrooms"] > 10].index, inplace = True)
df.drop(df[df["bedrooms"] == 0].index, inplace = True)
#df["bathrooms"] = np.array(df["bathrooms"])
df["bathrooms"] = np.around(df["bathrooms"])
df["bathrooms"] = df["bathrooms"].astype(int)

df["floors"] = np.around(df["floors"])
df["floors"] = df["floors"].astype(int)
pd.Series(df["floors"]).value_counts()
df["basement"] = df["sqft_basement"].apply(lambda x : 1 if x > 0 else 0)

df["date"] = pd.to_datetime(df["date"])
df["yr_sold"] = df["date"].dt.year

df["age_house"] = df["yr_sold"] - df["yr_built"]
df["any_renovation"] = df["yr_renovated"].apply(lambda x : 1 if x > 0 else 0)
df["sqft_living_change"] = (df["sqft_living15"] - df["sqft_living"])
df["sqft_living_change"] = df["sqft_living_change"].apply(lambda x : 0 if x == 0 else 1)
#df["sqft_living_change"]
df["sqft_living_inc"] = (df["sqft_living15"] - df["sqft_living"])
df["sqft_living_inc"] = df["sqft_living_inc"].apply(lambda x : 0 if x < 0 else x)
df["sqft_living_inc"]
df["sqft_living_dec"] = (df["sqft_living"] - df["sqft_living15"])
df["sqft_living_dec"] = df["sqft_living_dec"].apply(lambda x : 0 if x < 0 else x)
df["sqft_living_dec"]
df.head()
df["renovation_after_sold"] = (df["yr_sold"] - df["yr_renovated"]).apply(lambda x : 1 if x <= 0 else 0)
#df[df["renovation_after_sold"] == 1]
col_scatterplot = unique_df[unique_df["unique_count"] > 70]["features"]
col_boxplot = unique_df[unique_df["unique_count"] <= 70]["features"]

import matplotlib.pyplot as plt
for col in col_scatterplot:
    if col != "date":
        if col != "price": 
            plt.scatter(df[col], df["price"])
            plt.xlabel(col)
            plt.ylabel("Price")
            plt.show()

import seaborn as sns
for col in col_boxplot:
    if col != "price":
        sns.boxplot(df[col], df["price"])
        plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 12))
corr = df.corr()
sns.heatmap(corr, annot = True,fmt='.2f',
    cmap=sns.diverging_palette(20, 220, n=200),square=True)
x = df[["basement","yr_renovated","bathrooms", "bedrooms", "sqft_living","sqft_basement", "any_view", "no_of_views", "grade", "sqft_above", "sqft_living15", "floors", "waterfront","lat", "sqft_living_dec", "any_renovation"]]
y = df["price"]
from sklearn.model_selection import train_test_split as split_data
from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = split_data(x, y, random_state = 0)
reg = LinearRegression()
reg.fit(x_train, y_train)
predicted = reg.predict(x_test)
from sklearn import metrics
r_square = metrics.r2_score(y_test, predicted)
print(r_square)
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators=500,max_depth=4,min_samples_split=2,learning_rate=0.05,loss='ls')
regressor.fit(x_train, y_train)
GBM_predicted_Values=regressor.predict(x_test)
print('Gradient Boosting Regression R-squared', round(regressor.score(x_test, y_test), 3))
    