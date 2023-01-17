# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn 

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/housing.csv")



data.head()
data.info()
data.loc[data["total_bedrooms"].isna()]
data.groupby("ocean_proximity").describe()["median_house_value"]
data.plot(kind = "scatter", x="longitude", y="latitude", alpha=0.1)
data.plot(kind = "scatter", x="longitude", y="latitude", alpha=0.1, c="median_house_value")
data["total_bedrooms"].isna().sum()
# data.loc[data["total_bedrooms"].isna(),"total_bedrooms"] = data.loc[~data["total_bedrooms"].isna(),"total_bedrooms"].median()
# data.loc[~data["total_bedrooms"].isna(),"total_bedrooms"].median()





from sklearn.impute import SimpleImputer



data_numerical = data.drop("ocean_proximity",axis=1)

data_categorical = data["ocean_proximity"]



my_imputer = SimpleImputer(strategy="median")



my_imputer.fit(data_numerical)





data_numerical_transformed = pd.DataFrame(my_imputer.transform(data_numerical),columns=data_numerical.columns)
from sklearn.linear_model import LinearRegression



lr_model = LinearRegression()

lr_model.fit(data_numerical_transformed.drop("median_house_value",axis=1),data_numerical_transformed["median_house_value"])
lr_model.predict(data_numerical_transformed.drop("median_house_value",axis=1))