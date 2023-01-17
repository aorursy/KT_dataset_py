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
dt_store = pd.read_csv("../input/googleplaystore.csv")

dt_store.head()
dt_store.info()
print(dt_store.shape)
#data type cleaning

#clean reviews data

#We find one record containing 3.0M that leads us not being able to convert it right to numeric

dt_store.loc[dt_store.Reviews == '3.0M', 'Reviews'] = "3000000"

dt_store['Reviews'] = pd.to_numeric(dt_store['Reviews'])

#Besides, we find some price value is recorded as 'Everyone', I'm not sure what does it mean, here I simply convert all of them to 0

dt_store.loc[dt_store.Price == 'Everyone', 'Price'] = "0"

dt_store['Price'] = pd.to_numeric(dt_store['Price'].apply(lambda x: x.strip("$")))

dt_store.info()
dt_store.head()
#create a whole list of genres

new_genre = dt_store["Genres"].str.split(";", n = 1, expand = True) 

new_genre.head(100)

#create a full unique list of genres

first_list = list(new_genre[0].unique())

second_list = list(new_genre[1].unique())

in_first = set(new_genre[0].unique())

in_second = set(new_genre[1].unique())

in_second_but_not_in_first = in_second - in_first

full_genre_list = list(first_list) + list(in_second_but_not_in_first)



#for each genres, we create a dummy columns for them

for g in full_genre_list:

    dt_store[str(g)] = dt_store["Genres"].apply(lambda x: 1 if str(g) in x else 0)

dt_store.head(30)
#make a plot about the distribution of genre categories

import matplotlib.pyplot as plt

list(dt_store.columns)[13:]

g_dic = {}

for g in full_genre_list:

    a = {str(g): dt_store[str(g)].sum()}

    g_dic.update(a)

lists = sorted(g_dic.items(), key=lambda x: x[1])

# equiv# sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.rcParams["figure.figsize"] = (10,16)

plt.barh(x, y)

plt.title("Number of Apps Distribution across Genres",fontsize=15)

plt.show()
#free/paid distribution

type_c = dt_store.groupby(["Type"]).size()/dt_store.shape[0]

plt.rcParams["figure.figsize"] = (10,5)

type_c[1:].plot(kind='barh')

plt.title("Proportion of Paid/Free Apps",fontsize=15)

plt.show()
#Review Distribution

plt.hist(dt_store['Rating'],bins=20)
dt_store.loc[dt_store['Rating'] > 5.0, 'Rating'] = 5

plt.hist(dt_store['Rating'],bins=20)

print(dt_store['Rating'].describe())
#paid apps distribution

paid_app = dt_store.loc[dt_store['Type'] == 'Paid']



import matplotlib.pyplot as plt

list(dt_store.columns)[13:]

g_dic = {}

for g in full_genre_list:

    a = {str(g): paid_app[str(g)].sum()}

    g_dic.update(a)

lists = sorted(g_dic.items(), key=lambda x: x[1])

# equiv# sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.rcParams["figure.figsize"] = (10,16)

plt.barh(x, y)

plt.title("Number of Apps Distribution across Genres for Paid Apps",fontsize=15)

plt.show()
plt.rcParams["figure.figsize"] = (16,10)

plt.hist(paid_app.Price, bins=40)
paid_app.loc[paid_app['Price'] > 350].head()
#rate and review

plt.scatter(dt_store['Reviews'],dt_store['Rating'])

plt.ylim(top=5.5, bottom =0.5)

#plt.xlim(right=3000000, left=-1)

plt.rcParams["figure.figsize"] = (10,10)

plt.show()
#rate and review

plt.scatter(np.log(dt_store['Reviews']),dt_store['Rating'])

plt.ylim(top=5.5, bottom =0.5)

#plt.xlim(right=3000000, left=-1)

plt.rcParams["figure.figsize"] = (10,10)

plt.show()