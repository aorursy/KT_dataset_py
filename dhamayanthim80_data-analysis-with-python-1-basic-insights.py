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
import pandas as pd

import seaborn as sns

car_data = pd.read_csv("../input/automobile-from-california/imports-85.csv",header=None)
car_data.head()

car_data.tail()
# create headers list

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",

         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",

         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",

         "peak-rpm","city-mpg","highway-mpg","price"]

print("headers\n", headers)
car_data.columns = headers

car_data.head(10)
#car_data.dropna(subset=["price"],axis=0)

car_data.dropna(subset=["price"], axis=0)
car_data.head(10)
print(car_data.columns)
car_data.to_json("test.json")
#getting datatype of dataframe car_data by .dtypes

print(car_data.dtypes)
car_data.describe()
car_data.describe(include="all")
car_data[['make','price']]
car_data[['make','price']].describe()
print(car_data.info)