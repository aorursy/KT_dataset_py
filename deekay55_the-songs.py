# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/top50spotify2019/top50.csv", encoding = "ISO-8859-1")

del data["Unnamed: 0"]

data.columns=[title.replace(".","_") for title in data.columns]

data.columns = [title.split("_")[0].lower() if title.split("_")[-1]=="" else title.lower() for title in data.columns]

data.head()
data.info()
data.columns
data.corr()
data.length.plot(kind="line", color="red", figsize=(10,10), grid=True, title="Length of Best Songs", fontsize=17, linewidth=3)

plt.xlabel("Songs")

plt.ylabel("Length")

plt.legend()

plt.show()
data.beats_per_minute.plot(kind="hist", bins=50, figsize=(10,10), grid=True, fontsize=17, title="Beats of Best Songs")

plt.show()
data.plot(kind="scatter", x="energy", y="danceability", color="r", linewidth=5, grid=True)

plt.xlabel("Energy")

plt.ylabel("Danceability")

plt.show()
data.plot(kind="scatter", x="energy", y="loudness", color="r", linewidth=5, grid=True)

plt.xlabel("Energy")

plt.ylabel("Loudness (dB)")

plt.show()
data.popularity.plot(kind="line", color="red", figsize=(10,10), grid=True, title="Popularity of Best Songs", fontsize=17, linewidth=3)

plt.xlabel("Songs")

plt.ylabel("Popularity")

plt.show()
print(data.genre.value_counts())
# Artists with more than 1 song in the best songs

a=data.artist_name.value_counts()

x=a>1

print(a[x])