# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

from plotly import tools

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px 

from wordcloud import WordCloud 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_all = pd.read_csv("/kaggle/input/top-spotify-tracks-of-2018/top2018.csv")
data_all.head()
data_all.info()
data_all_corr = data_all.corr()

data_all_corr



ax = sns.heatmap(data_all_corr,cmap="YlGnBu",annot = False)
import plotly.express as px 

fig = px.scatter(data_all_corr,x = 'danceability',y = 'energy' )

fig.show()
speechiness_mean = np.mean(data_all.speechiness)
above_speec = data_all['speechiness'] > speechiness_mean

speec_above = data_all[above_speec]

speec_above = speec_above.sort_values(by = ['speechiness'])

speec_above = speec_above[::-1]

speec_above[["name","speechiness"]].head(10)







data_all.head()
from wordcloud import WordCloud 

artist_data = data_all.artists

artist_sum = artist_data.value_counts()

max_artist_space = artist_sum.index[:7]

max_artist=[]



for i in max_artist_space:

    j = i.replace(' ','')

    max_artist.append(j)

plt.subplots(figsize = (8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate("/".join(max_artist))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

    

data_all_pie = data_all.artists

data_all_pie.value_counts()

labels = data_all_pie.value_counts().index

explode = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

size = data_all_pie.value_counts().values

plt.figure(figsize = (25,25))

plt.pie(size, explode=explode, labels=labels,autopct='%1.1f%%')

plt.title('Artist Dominance',color='black')



sns.jointplot(data_all_corr.tempo, data_all_corr.energy, kind="hex")

plt.show()

sns.set(style="white")

sns.jointplot(data_all_corr.liveness, data_all_corr.instrumentalness, kind="kde", height=7, space=0)
data_all_corr.head(13)
data_all_corr.speechiness

data_all_corr.acousticness

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

sns.set(style="white")

sns.relplot(x="instrumentalness", y="acousticness", size="speechiness",

            sizes=(40, 400), alpha=.5, palette="colors",

             data=data_all_corr)