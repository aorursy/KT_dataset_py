# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



pd.set_option("display.max_columns", None)

plt.style.use("ggplot")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
songs = pd.read_csv("../input/top50.csv", encoding="latin")
print(songs.info())

print(songs.head())
mean_seconds = songs["Length."].mean()

mean_minutes = round(mean_seconds / 60, 2)

print(mean_minutes)
singers = songs["Artist.Name"].value_counts()

singers.plot.bar()

plt.xlabel("Singers")

plt.ylabel("# of Songs")

plt.title("The # of songs each singer has")

plt.show()
genres = songs["Genre"].value_counts()

genres.plot.bar()

plt.xlabel("Genres")

plt.ylabel("# of Songs")

plt.title("The # of songs each genre has")

plt.show()
singer_popularity = (

    songs.groupby("Artist.Name")["Popularity"].sum().sort_values(ascending=False)

)

singer_popularity.plot.bar()

plt.xlabel("Singers")

plt.ylabel("Total popularity")

plt.title("Total popularity each singer has")

plt.show()
genre_popularity = (

    songs.groupby("Genre")["Popularity"].sum().sort_values(ascending=False)

)

genre_popularity.plot.bar()

plt.xlabel("Genres")

plt.ylabel("Total popularity")

plt.title("Total popularity each genre has")

plt.show()
plt.scatter("Danceability", "Popularity", data=songs.sort_values(by=["Danceability"]))

plt.title("The relationship between danceability and popularity")

plt.xlabel("Danceability")

plt.ylabel("Popularity")

plt.show()
plt.scatter(

    "Loudness..dB..", "Popularity", data=songs.sort_values(by=["Loudness..dB.."])

)

plt.title("The relationship between dB and popularity")

plt.xlabel("dB")

plt.ylabel("Popularity")

plt.show()
plt.scatter("Liveness", "Popularity", data=songs.sort_values(by=["Liveness"]))

plt.title("The relationship between liveness and popularity")

plt.xlabel("Liveness")

plt.ylabel("Popularity")

plt.show()