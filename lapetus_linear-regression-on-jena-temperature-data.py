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
dirname = "/kaggle/input/weather-archive-jena"

filename = "jena_climate_2009_2016.csv"

filename = os.path.join(dirname, filename)

jena = pd.read_csv(filename)

jena.info()
jena.describe()
jena.head(10)
%matplotlib inline
import matplotlib.pyplot as plt

jena.hist(bins=50, figsize=(20, 15))
plt.show()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(jena, test_size=0.2, random_state=42)
jena = train_set.copy()
jena["Date Time"].value_counts()
from datetime import datetime

jena["Date Time"] = jena["Date Time"].astype("datetime64[s]")

jena["Hour"] = jena["Date Time"].dt.hour
jena["Year"] = jena["Date Time"].dt.year
jena["Month"] = jena["Date Time"].dt.month
jena["Day_of_Year"] = jena["Date Time"].dt.dayofyear

jena.info()

jena.describe()
correlation_matrix = jena.corr()
correlation_matrix["T (degC)"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

selected_attributes = ["T (degC)", "VPmax (mbar)", "Month", "rho (g/m**3)"]
scatter_matrix(jena[selected_attributes], figsize=(15, 10))
# seasons encoding
# 0 - winter, 1 - spring, 2 - summer, 3 - autumn
jena["Season"] = ((jena["Month"]%12 + 3)//3) - 1
#jena.head(25)
correlation_matrix1 = jena.corr()
correlation_matrix1["T (degC)"].sort_values(ascending=False)
# Separating features from labels
jena = train_set.drop("T (degC)", axis=1).copy()
jena_labels = train_set["T (degC)"].copy()
from sklearn.base import BaseEstimator, TransformerMixin

datetime_ix = 0

class SeasonAttribsAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_seasons_encoding=True):
        self.add_seasons_encoding = add_seasons_encoding
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X["Season"] = (X["Date Time"].astype("datetime64[s]").dt.month % 12 + 3)//3 - 1
        return X["Season"].values.reshape(1, -1)
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values
# Creating a numerical pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_attribs = list(train_set.drop("Date Time", axis=1).copy())

num_pipeline = Pipeline([
                            ("selector", DataFrameSelector(num_attribs))
                            ("feature-scaling", StandardScaler())
                        ])

# transformed dataset
jena_tr = num_pipeline.fit_transform(jena)
# Create a categorical pipeline

from sklearn.preprocessing import OneHotEncoder

cat_attribs = ["Season"]

cat_pipeline = Pipeline([
                        ("convert-to-seasons", SeasonAttribsAdder()),
                        ("cat-encodeer", OneHotEncoder())
                        ])

jena = train_set.copy()

jena_tr = cat_pipeline.fit_transform(jena)
