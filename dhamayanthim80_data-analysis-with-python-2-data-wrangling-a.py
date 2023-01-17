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
# create headers list

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",

         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",

         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",

         "peak-rpm","city-mpg","highway-mpg","price"]

print("headers\n", headers)
car_data.columns = headers

car_data.head(10)
import numpy as np

car_data.replace("?", np.nan, inplace = True)

car_data.head(10)
missing_data = car_data.isnull()

missing_data.head(10)
for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print("")

    
car_data.describe(include = "all")
car_data['num-of-doors'].value_counts()
car_data['num-of-doors'].value_counts().idxmax()

car_data['num-of-doors'].replace(np.nan, "four", inplace=True)

missing_data_1 = car_data.isnull()

print(missing_data_1['num-of-doors'].value_counts())

car_data.dropna(subset=["price"], axis = 0, inplace=True)

# reset index, because we droped two rows

car_data.reset_index(drop=True, inplace=True)

missing_data_2 = car_data.isnull()

print(missing_data_2['price'].value_counts())
avg_norm_loss = car_data["normalized-losses"].astype("float").mean(axis=0)

avg_stroke =car_data["stroke"].astype("float").mean(axis=0)

avg_bore = car_data["bore"].astype("float").mean(axis=0)

avg_horse_power = car_data["horsepower"].astype("float").mean(axis=0)

avg_peak_rpm = car_data["peak-rpm"].astype("float").mean(axis=0)

print(avg_norm_loss, "" , avg_stroke, "", avg_bore, "", avg_horse_power, "", avg_peak_rpm)
car_data["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

car_data["stroke"].replace(np.nan, avg_stroke, inplace=True)

car_data["bore"].replace(np.nan, avg_bore, inplace=True)

car_data["horsepower"].replace(np.nan, avg_horse_power, inplace=True)

car_data["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace=True)
car_data.head(10)

car_data.tail(20)
#list data types of each column

car_data.dtypes
car_data[["bore", "stroke", "price", "peak-rpm"]] = car_data[["bore", "stroke", "price", "peak-rpm"]].astype("float")

car_data[["normalized-losses"]] = car_data[["normalized-losses"]].astype("int")

car_data.dtypes

#check data of the coulms engine-type and num-of-cylinders

df = car_data[["engine-type","num-of-cylinders"]]

df