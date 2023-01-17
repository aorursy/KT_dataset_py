import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from wordcloud import WordCloud



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/top-spotify-songs-from-20102019-by-year/top10s.csv", encoding='ISO-8859-1') 
data.head()
# Drop the unnamed column



data.drop('Unnamed: 0', axis=1, inplace=True)
data.columns
msno.matrix(data)
# Renaming few columns to make more sense



data.rename(columns = {'top genre': 'top_genre', 'bpm': 'beats_per_minute', 'nrgy': 'energy', 

                       'dnce': 'danceability', 'dB': 'loudness(dB)', 'live': 'liveness', 

                       'val': 'valence', 'dur': 'length', 'acous': 'acousticness', 

                       'spch': 'speechiness', 'pop': 'popularity'}, inplace=True)
data.dtypes
corr = data.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 14))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.lineplot(x="energy", y="acousticness", data=data)
sns.catplot(y="liveness", x="loudness(dB)", kind="swarm", data=data)
sns.catplot(y="beats_per_minute", x="year", kind="boxen", data=data)
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(data.danceability, data.popularity, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x=data.speechiness, y=data.length, data=data, kind="kde", color="k")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Speechiness$", "$Length$")
sns.jointplot(x=data.valence, y=data.popularity, data=data, kind="kde");
wordcloud = WordCloud(width = 1000, height = 600, max_font_size = 200, max_words = 150,

                      background_color='white').generate(" ".join(data.top_genre))



plt.figure(figsize=[10,10])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
data['top_genre'].value_counts().head(10).plot.pie(figsize=(15,10), autopct='%1.0f%%')
plt.figure(figsize=(16,8))

plt.title('Most frequent Artist',fontsize=30)

plt.xlabel('Artist', fontsize=20)

plt.ylabel('Count', fontsize=20)



sns.countplot(data.artist,order=pd.value_counts(data.artist).iloc[:15].index,palette=sns.color_palette("coolwarm", 15))



plt.xticks(size=20,rotation=90)

plt.yticks(size=20)

sns.despine(bottom=True, left=True)

plt.show()
plt.figure(figsize=(16,8))

plt.title('Most frequent Titles',fontsize=30)

plt.xlabel('Title', fontsize=25)

plt.ylabel('Count', fontsize=25)



sns.countplot(data.title,order=pd.value_counts(data.title).iloc[:25].index,palette=sns.color_palette("magma", 25))



plt.xticks(size=20,rotation=90)

plt.yticks(size=20)

sns.despine(bottom=True, left=True)

plt.show