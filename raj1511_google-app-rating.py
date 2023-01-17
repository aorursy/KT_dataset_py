# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
google_data = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
type(google_data)
google_data.head()     # Inspecting first 5 Rows
google_data.shape
google_data.describe()    # Summary Stats
google_data.boxplot()
google_data.hist()
google_data.info()
google_data.isnull()
# Count the number of missing value in each column



google_data.isnull().sum()
google_data[google_data.Rating > 5]
google_data.drop([10472], inplace=True)
google_data[10470:10475]
google_data.boxplot()
google_data.hist()
threshold = len(google_data)*0.1

threshold
google_data.dropna(thresh = threshold, axis = 1, inplace = True)
print(google_data.isnull().sum())
# Define a function impute_median



def impute_median(series):

    return series.fillna(series.median())

google_data.Rating = google_data["Rating"].transform(impute_median)
# count the number of null values in each column

google_data.isnull().sum()
# mode of categorical values



print(google_data["Type"].mode())

print(google_data["Current Ver"].mode())

print(google_data["Android Ver"].mode())
# fill the missing categorical values with mode



google_data["Type"].fillna(str(google_data["Type"].mode().values[0]),inplace = True)

google_data["Current Ver"].fillna(str(google_data["Current Ver"].mode().values[0]),inplace = True)

google_data["Android Ver"].fillna(str(google_data["Android Ver"].mode().values[0]),inplace = True)
# count the number of null values in each column

google_data.isnull().sum()
# let's convert Price, Reviews and ratings into numerical values



google_data["Price"] = google_data["Price"].apply(lambda x : str(x).replace('$', '') if '$' in str(x) else str(x))

google_data["Price"] = google_data["Price"].apply(lambda x : float(x))

google_data["Reviews"] = pd.to_numeric(google_data["Reviews"], errors = 'coerce')

google_data["Installs"] = google_data["Installs"].apply(lambda x : str(x).replace('+', '') if '+' in str(x) else str(x))

google_data["Installs"] = google_data["Installs"].apply(lambda x : str(x).replace(',', '') if ',' in str(x) else str(x))

google_data["Installs"] = google_data["Installs"].apply(lambda x : float(x))
google_data.head(10)
# summary stats after cleaning

google_data.describe()