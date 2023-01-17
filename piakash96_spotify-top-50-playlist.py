# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/top50spotify2019/top50.csv", encoding = "ISO-8859-1", index_col = "Unnamed: 0")
data.head()
data.columns
columns_to_check = ['Beats.Per.Minute', 'Energy', 'Danceability', 'Loudness..dB..', 'Liveness', 'Valence.', 

                    'Length.','Acousticness..', 'Speechiness.', 'Popularity']



for columns in columns_to_check:

    proprty = {"genre" : [], "property" : []}

    

    for genre in data["Genre"].value_counts().index:

        proprty["genre"].append(genre)

        proprty["property"].append(data[data["Genre"] == genre][columns].mean())

    

    proprty = pd.DataFrame(proprty)

    proprty = proprty.sort_values(by = ["property"], ascending = False)

    

    # plotting the average property for distinct genres:

    plt.figure(figsize = (15,5))

    sns.barplot(proprty["genre"], proprty["property"])

    plt.xticks(rotation = 45)

    plt.ylabel("{}".format(columns))

    plt.title("Average {} with respect to distinct genres in the playlist".format(columns))

    plt.show()
temp = data[["Artist.Name", "Genre"]]
# plot showing relationship between the artists and the genres

plt.figure(figsize = (15,15))

sns.heatmap(pd.pivot_table(temp, columns = "Artist.Name", index = "Genre", aggfunc = len, fill_value = 0), square = True, annot = True)

plt.show()
# table showing the threshold we take to identify the 75% threshold so that we can identify the top genres

data["Popularity"].describe()
# filtering out the songs from playlist which has the popularity index greater than the 90.75 threshold

popular_genres = data[data["Popularity"] > 90.75]
popular_genres
fig = plt.figure(figsize = (20,10))



plt.subplot(1,2,1)

temp = popular_genres[["Artist.Name", "Genre"]]

sns.heatmap(pd.pivot_table(temp, columns = "Artist.Name", index = "Genre", aggfunc = len, fill_value = 0), square = True, annot = True)



plt.subplot(1,2,2)

popular = {"genre" : [], "popular" : []}

    

for genre in popular_genres["Genre"].value_counts().index:

    popular["genre"].append(genre)

    popular["popular"].append(popular_genres[popular_genres["Genre"] == genre]["Popularity"].mean())



popular = pd.DataFrame(popular)

popular = popular.sort_values(by = ["popular"], ascending = False)



# plotting the average property for distinct genres:

sns.barplot(popular["genre"], popular["popular"])

plt.xticks(rotation = 45)

plt.ylabel("{}".format(columns))

plt.title("Average {} with respect to distinct genres in the playlist".format(columns))

plt.ylim(80,100)



plt.show()