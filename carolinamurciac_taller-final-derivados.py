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
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats

from sklearn import preprocessing

import os

print(os.listdir("../input"))

print(os.chdir("../input"))

import pandas as pd 

my_data= pd.read_csv('../input/top50spotify2019/top50.csv', encoding = "ISO-8859-1")
my_data.head()
my_data[["Artist.Name","Track.Name"]].groupby("Artist.Name").count() 



nca=my_data.groupby(["Artist.Name"]).count().iloc[:,:1]

nca.columns=['Number of Songs']

print(nca)
import matplotlib.pyplot as plt

from itertools import cycle, islice

df = pd.DataFrame(nca)

fig, ax = plt.subplots()

fig.set_size_inches(6,6)

df.plot.barh(stacked=True, ax=ax, color='pink')

ax.set_title("Número de canciones por artista")

plt.xlabel("Nombre Artista")

plt.xlabel("Número de canciones")





my_data[["Genre","Track.Name"]].groupby("Genre").count()
npg=my_data.groupby(["Genre"]).count().iloc[:,:1]

npg.columns=['Number of Genre']

print(npg)
genre = my_data["Genre"].value_counts()

genre.plot.bar()

plt.xlabel("Género")

plt.ylabel("Número de canciones")

plt.title("Género por Canciones")



my_data[["Artist.Name","Popularity"]]
data = my_data["Popularity"]

fig, axs = plt.subplots()

axs.boxplot(data)

axs.set_title('Popularidad de las Canciones')



my_data['Popularity'].quantile([.25, .5, .75])
my_data[["Track.Name","Beats.Per.Minute"]]
sns.distplot(my_data["Beats.Per.Minute"],  color="green")

plt.title("Beats por Canción")

my_data[["Track.Name","Energy","Danceability"]]
plt.plot(my_data["Danceability"], color="r")  

plt.xlabel("Top de canciones")

plt.ylabel("Valor")

plt.ioff()   

plt.plot(my_data["Energy"], color="y") 

plt.ion()

plt.title("Energía por Canción vs si es Bailable")

plt.legend()

  
my_data[["Track.Name","Popularity","Danceability"]]
plt.scatter("Danceability", "Popularity", data=my_data[["Track.Name","Popularity","Danceability"]], color="c")

plt.title("¿Si la canción es bailable es más popular?")

plt.xlabel("Bailable")

plt.ylabel("Popularidad")

plt.show()