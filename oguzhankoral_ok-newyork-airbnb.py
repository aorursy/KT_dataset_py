# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.columns
data.info()
data.corr()
# correlation map

f,ax = plt.subplots(figsize = (18,18))

sns.heatmap(data.corr(), annot=True, linewidths=5, fmt=".1f", ax=ax)

plt.show()
data.head(10)
data.price.plot(kind = 'line', color = 'g', label = 'Price', linewidth=1, alpha = 0.5, grid = True, linestyle = ':',figsize = (12,12))

data.number_of_reviews.plot(color = 'r', label = 'Price', linewidth=1, alpha = 0.5, grid = True, linestyle = '-.')

plt.legend(loc='upper left')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
print(data['minimum_nights'].value_counts(dropna =False))
data.describe()
datafiltered = data["price"]<2000
data[datafiltered]
x = data["price"]<200

data[x]
# scatter plot

# x = price, y = reviews_per_month

data[x].plot(kind="scatter", x ="price", y="reviews_per_month", alpha = 0.5, color = "r",figsize = (12,12))

plt.xlabel("Price")

plt.ylabel("Reviews per Month")

plt.title("Price & Activeness")
# Histogram

# bins = number of bar in figure

data.price.plot(kind="hist", bins = 50, figsize = (12,12))

plt.show()
dictionary = {"uskudar" : "icadiye",

             "kadikoy" : "moda"}

print(dictionary.keys())

print(dictionary.values())

print(dictionary)
dictionary["uskudar"] = "kuzguncuk", "icadiye"

print(dictionary)

dictionary["kadikoy"] = "moda", "fikirtepe"

print(dictionary)

print("kadikoy" in dictionary)
dictionary.clear()
data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
x = data["price"]<2000

data[x]
data[np.logical_and(data["price"]<200, data["reviews_per_month"]>20)]
data[(data["price"]<200) & (data["reviews_per_month"]>20)]
series = data["price"]

print(type(series))

data_frame = data[["price"]]

print(type(data_frame))

series

i = 0

while i != 5 :

    print("i is: ", i)

    i += 1

print(i, " is equal to 5")
liste = [1,2,3,4,5,6,7,8]

for i in liste:

    print("i is: ",i)

print("")



dictionary = {"uskudar": ["icadiye","kuzguncuk"],"kadikoy": ["moda", "goztepe"]}

for index,value in dictionary.items():

    print(index, " : ", value)

    

for index,value in data[["price"]][0:10].iterrows():

    print(index, " : ",value)
# For example lets look frequency of pokemom types

print(data['room_type'].value_counts(dropna =False))  # if there are nan values that also be counted

# As it can be seen below there are 112 water pokemon or 70 grass pokemon
data.boxplot(column="price", by ="room_type",figsize = (18,18))

plt.show()
data_new = data.head()

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = "host_name", value_vars = ["price","reviews_per_month"])

melted
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row
data3 = data['price'].head()

data4 = data['reviews_per_month'].head()

conc_data_col = pd.concat([data3,data4],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
data["latitude"] = data["latitude"].astype("int")
data.dtypes
data.head()
data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.head()
data.info()
data["last_review"].value_counts(dropna=False) # dropna = false be useful to see NaN objects
data1 = data # when we change the values of the data1, It will aşsp change in the data as you can see on the 3rd row code

data1["last_review"].dropna(inplace=True) # drop the NaN objects in the data1 and rewrite on it

data1["last_review"].value_counts(dropna=False)
assert data["last_review"].notnull().all()

# it will return nothing
data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
assert data["last_review"].notnull().all()

# it will return error
assert data.price.dtypes == np.int

# it will return nothing because type of price column is int