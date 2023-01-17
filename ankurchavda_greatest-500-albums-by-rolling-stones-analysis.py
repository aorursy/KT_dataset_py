import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud, STOPWORDS
import os

#import file to dataframe
df_album = pd.read_csv("../input/albumlist.csv", encoding="ISO-8859-1")
#View data
df_album.head(5)
print(df_album.describe())
df_album.info()
# Checking for null values
df_album.isna().any()
# Number of unqiue artists in the data
df_album['Artist'].nunique()
order = df_album['Artist'].value_counts(ascending=False).head(50).index
plt.figure(figsize=(15,10))
ax = sns.countplot('Artist', data = df_album , order= order)
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
ax.set_title(label="Most number of Albums by Bands",fontdict={'fontsize':15})
ax.set(ylabel='Albums')
plt.tight_layout()
plt.show()
# order = df_album['Year'].value_counts(ascending=False).head(50).index
plt.figure(figsize=(15,10))
ax = sns.countplot('Year', data = df_album , palette='GnBu_d')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
ax.set_title(label="Most number of Albums by Year",fontdict={'fontsize':15})
ax.set(ylabel='Albums')
plt.tight_layout()
plt.show()
#data cleanup and split genres
df_split = (df_album.set_index(df_album.columns.drop('Genre',1).tolist()).Genre.str.split(',', expand=True).stack().reset_index().rename(columns={0:'Genre'}).loc[:, df_album.columns])
df_split.replace({'Genre':{' & Country':'Country', 'ÊPop':'Pop', 'ÊFolk': 'Folk', 'ÊStage & Screen':'Stage & Screen','ÊBlues':'Blues'}}, inplace= True)
df_split['Genre'] = df_split['Genre'].str.strip()
df_split.head()
plt.figure(figsize=(15,8))
order = df_split['Genre'].value_counts(ascending=False).index
ax = sns.countplot(y='Genre', data = df_split, order= order,  palette='BrBG')
ax.set_title(label="Most popular music genres",fontdict={'fontsize':15})
ax.set(xlabel='# of times included in album genres')
plt.show()
year = df_split['Year']

def to_decade(value):
    return int(value/10)*10

df_split['Decade'] = year.apply(to_decade)
plt.figure(figsize=(15,8))
ax = sns.countplot(x='Decade', data = df_split, palette='GnBu_d')
ax.set_title(label='Number of Albums in each decade',fontdict={'fontsize':15})
ax.set(ylabel='Albums')
# df_split.groupby('Year').Genre.count().
plt.figure(figsize=(22,14))
ax = sns.stripplot(x='Year', y='Genre', data= df_split, jitter= 0.15, size= 10)
ax.set_title(label='Most popular genre from each year',fontdict={'fontsize':22})
plt.show()
#data cleanup and split genres
df_subgen = (df_split.set_index(df_split.columns.drop('Subgenre',1).tolist()).Subgenre.str.split(',', expand=True).stack().reset_index().rename(columns={0:'Subgenre'}).loc[:, df_split.columns])
df_subgen.Subgenre.replace({'Ê':''}, regex=True, inplace=True)
df_subgen.Subgenre.replace({'Musique Concr?te':'Musique Concrete'}, inplace=True)
df_subgen['Subgenre'] = df_subgen['Subgenre'].str.strip()
df_subgen.Subgenre.unique()
# Influnce of each subgenre in a genre
genres = df_subgen['Genre'].unique().tolist()
num_of_genres = (len(genres))

fig = plt.figure(
        figsize = (50, 50),
        facecolor = 'w',
        edgecolor = 'k')
fig.tight_layout()
fig.suptitle('Most popular sub genres in each genre',fontsize=70, y=1.05)
for i,v in enumerate(range(num_of_genres)):
    v = v+1
    text = df_subgen[df_subgen['Genre'] == genres[i]]['Subgenre']
    text = text.str.cat(sep=' ')
    wordcloud = WordCloud(
        width = 400,
        height = 350,
        background_color = 'white',
        stopwords = STOPWORDS).generate(text)

    ax1 = plt.subplot(int(num_of_genres/2) + 1,4,v)
    ax1.set_title(genres[i], fontdict={'fontsize':50})
    ttl = ax.title
    ttl.set_position([.9, 1.05])
    plt.subplots_adjust(hspace = 1)
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
plt.show()