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
data = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")
data.info()
data.head()
data1 = data.head()

data2 = data.tail()

data3 = pd.concat([data1,data2], axis = 0 , ignore_index = True)

data3
data.columns
data["genres"].value_counts(dropna =False).head(10)
data["original_language"].value_counts(dropna = False)
# id_vars = what we do not wish to melt

# value_vars = what we want to melt

data_new = data.head()

melted = pd.melt(frame=data_new,id_vars = 'title', value_vars= ['vote_average','vote_count'])

melted
data['popularity'].describe()
data0 = data['title']

data1 = data['popularity']

data2= data['vote_average']

conc_data_col = pd.concat([data0,data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col1 =conc_data_col.sort_values('popularity', ascending=False)

conc_data_col1
plt.figure(figsize=(12,4))

plt.barh(conc_data_col1['title'].head(10),conc_data_col1['popularity'].head(10), align='center',color='skyblue')

plt.gca().invert_yaxis()

plt.xlabel("Popularity")

plt.title("Popular Movies")

plt.show()
movies = data.sort_values('vote_average', ascending=False)

movies[['title', 'vote_count', 'vote_average']].head(40)
datetime_object = pd.to_datetime(data["release_date"])

print(type(datetime_object))