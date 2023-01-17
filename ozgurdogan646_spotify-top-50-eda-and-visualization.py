# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import figure



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/top50spotify2019/top50.csv",encoding="ISO-8859-1")
data.head()
data.describe()
data.shape
data.columns
data.rename(columns = { "Unnamed: 0" : "id",

                        "Acousticness.." : "Acousticness",

                        "Track.Name" : "Track_Name" ,

                        "Valence." : "Valence",

                        "Length." : "Length",

                        "Loudness..dB.." : "Loudness_dB" ,

                        "Artist.Name" : "Artist_Name",

                        "Beats.Per.Minute" :"Beats_Per_Minute",

                        "Speechiness." : "Speechiness"},inplace = True)
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.loc[:,'Artist_Name'].value_counts()
data.columns
#line plot

columns = ['Danceability','Valence','Length']

colors = ['g','b','r','y']

num = 0

for i,col in enumerate(columns):

    num+=1

    figure(figsize = (5,5))

    data["{}".format(col)].plot(kind = "line" , color =colors[i] , label = col , linestyle = '-')

    plt.legend()

    plt.xlabel("x axis")

    plt.xlabel("y axis")

    plt.title("{} Line Plot".format(col))

    plt.show()

    

#scatter plot

data.plot(kind = "scatter", x = "Popularity" , y = "Danceability" , color = "red")

plt.xlabel("Popularity")

plt.ylabel("Danceability")

plt.title("Popularity/Danceability scatter plot")

plt.show()
data.plot(kind = "scatter", x = "Popularity" , y = "Length" , color = "blue")

plt.xlabel("Popularity")

plt.ylabel("Length")

plt.title("Popularity/Length scatter plot")

plt.show()
#pie_graph

plt.figure(1, figsize=(8,8))

data.Artist_Name.value_counts().plot.pie(autopct="%1.1f%%")
data.columns
some_artist = ('Lil Nas X', 'Billie Eilish', 'Sech', 'Ariana Grande')



data_artist = data.loc[data['Artist_Name'].isin(some_artist) & data['Popularity']]



plt.rcParams['figure.figsize'] = (15, 8)

ax = sns.boxplot(x = data_artist['Artist_Name'], y = data_artist['Popularity'], palette = 'inferno')

ax.set_xlabel(xlabel = 'Some Popular Artists', fontsize = 9)

ax.set_ylabel(ylabel = 'Popularity\'s', fontsize = 9)

ax.set_title(label = 'Distribution of Popularities for some artists', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()