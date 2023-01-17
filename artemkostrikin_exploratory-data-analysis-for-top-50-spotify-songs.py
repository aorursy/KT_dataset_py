# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
sns.set(style="darkgrid")
import statistics as stat
import plotly.express as px
spotify = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding='ISO-8859-1')
spotify.head()
spotify.shape
spotify.info()
spotify_int = spotify.iloc[:, 4:14]
spotify_int.head()
spotify_int.describe()
# Plot correlation of all integer variables
sns.pairplot(spotify_int);
# shows bar gra aph of Popularity and Track Name
spotify.plot(y='Popularity',x= 'Track.Name',kind='bar',figsize=(26,6),legend =True,title="Popularity Vs Track Name",
        fontsize=18,stacked=True,color=['y', 'r', 'b','y', 'r', 'b', 'y'])
plt.ylabel('Popularity', fontsize=18)
plt.xlabel('Track Name', fontsize=18)
plt.show()
# Count of artist name
plt.figure(figsize=(10,10))
sns.countplot(y='Artist.Name', data=spotify, order=spotify["Artist.Name"].value_counts().index)
plt.show()
# Count by Genre
spotify['Genre'].value_counts().plot.bar()
plt.title('Count by Genre')
plt.ylabel('quanity')
plt.show()
print(spotify.groupby('Genre').size())
# Create wordcloud based on music genre
from wordcloud import WordCloud, STOPWORDS
# Create the wordcloud object
wordcloud = WordCloud(width=700, height=600, margin=3).generate(str(spotify.Genre))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.figure(figsize=(10,5))
swarmplot=sns.swarmplot(x='Genre',y='Popularity',data=spotify, s=13)
swarmplot.set_xticklabels(swarmplot.get_xticklabels(),rotation=90)
swarmplot.set_title('Relationship between Genre & Popularity')
# Visualization the relationship between Beats Per Minute and artists based on Popularity
sns.catplot (x = "Artist.Name", y = "Beats.Per.Minute", hue = "Popularity", s = 15, data = artist, kind = "swarm")
sns.catplot(x = "Loudness..dB..", y = "Energy", kind = "box", data = spotify)
# Spearman correlation statistics for all integer variables
pd.set_option('precision', 3)
corr = spotify.corr(method='spearman')
print(corr)
# Marginal plot between Acousticness and Beat Per Minute
sns.jointplot(x="Beats.Per.Minute", y="Acousticness..", data=spotify, kind="kde");
# Script for filter several artists
artist = spotify[spotify["Artist.Name"].isin(["Ed Sheeran", "J Balvin", "Ariana Grande", "Marshmello", "The Chainsmokers", "Shawn Mendes"])]