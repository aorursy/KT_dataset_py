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
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
import matplotlib.pyplot as plt

import seaborn as sb

import plotly.express as px



%matplotlib inline
#df.style.background_gradient(cmap='viridis')

df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
sb.heatmap(df.isnull(), cbar=False)
df.date_added = pd.to_datetime(df.date_added, infer_datetime_format=True)
df.head()
plt.rcParams["figure.figsize"] = [16,9]

#plt.figure(figsize=(15,4))

sb.countplot(x='release_year', hue='type', data = df)

plt.xticks(rotation=90)

df_plot = df[df.release_year >= 2000].groupby(['release_year', 'type'])['show_id'].count().reset_index()



fig = px.bar(df_plot, x="release_year", y="show_id", color="type", barmode="group", text='show_id')

fig.show()
df_t = df

df_t['year_added'] = df.date_added.dt.year

df_plot = df_t[df_t.year_added >= 2000].groupby(['year_added', 'type'])['show_id'].count().reset_index()



fig = px.bar(df_plot, x="year_added", y="show_id", color="type", barmode="group", text='show_id')

fig.show()
sb.lmplot(data = df_plot, x='year_added', y='show_id', hue='type')
df['rating'].value_counts()

df_plot = df.groupby(['rating', 'type'])['show_id'].count().reset_index()



fig = px.bar(df_plot, x="rating", y="show_id", color="type", barmode="group", text='show_id')

fig.show()
df.query("release_year == 2019")
fig = px.scatter(df.loc[df.release_year == 2018].dropna(), x="director", y="rating", color="country",

           hover_name="country",size_max=60)

fig.show()
fig = px.scatter(df.loc[df.release_year == 2019].dropna(), x="director", y="rating", color="country",

           hover_name="country", size_max=60)

fig.show()
import plotly.express as px

pxdf = px.data.gapminder()



country_isoAlpha = pxdf[['country', 'iso_alpha']].drop_duplicates()

country_isoAlpha.set_index('country', inplace=True)

country_map = country_isoAlpha.to_dict('index')
def getCountryIsoAlpha(country):

    try:

        return country_map[country]['iso_alpha']

    except:

        return country
df['iso_alpha'] = df['country'].apply(getCountryIsoAlpha)

df.head()
df_plot = df.groupby('iso_alpha').count().reset_index()

fig = px.choropleth(df_plot, locations="iso_alpha",

                    color="show_id", 

                    hover_name="iso_alpha", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()

df.drop('iso_alpha', axis=1, inplace=True)

df.loc[:, ['title', 'type', 'release_year', 'description']].sort_values(by='release_year')
df.loc[:, ['title', 'type', 'release_year', 'description']].sort_values(by='release_year', ascending=False)
def extractValues(listOfVals):

    values = []

    for val in listOfVals:

        for eachVal in val.split(", "):

            values.append(eachVal)

    return values
from collections import Counter



directors = extractValues(df.director.dropna())

director_counts = Counter(directors).most_common(30)

#print(list(director_counts[0].elements()))



directormap= {}

for k,v in director_counts:

    directormap[k] = v
directormap
df_dir = pd.DataFrame(directormap.items(), columns=['Director', 'No.ofShows'])

df_dir
px.bar(df_dir, x='Director', y='No.ofShows')
from collections import Counter



generes = extractValues(df.listed_in.dropna())

genere_counts = Counter(generes).most_common(15)

#print(list(director_counts[0].elements()))



generemap= {}

for k,v in genere_counts:

    generemap[k] = v
plt.figure(figsize=(8,5))

x = list(generemap.keys())

y = list(generemap.values())

sb.scatterplot(x=x, y=y, size=y, hue=y, palette="Set2", sizes=(25, 250))

plt.xticks(rotation=90)
'''

fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",

           size="pop", color="continent", hover_name="country", facet_col="continent",

           log_x=True, size_max=45, range_x=[100,100000], range_y=[25,90])

'''
from collections import Counter



casts = extractValues(df.cast.dropna())

cast_counts = Counter(casts).most_common(30)

#print(list(director_counts[0].elements()))



castmap= {}

for k,v in cast_counts:

    castmap[k] = v
plt.figure(figsize=(15,8))

sb.barplot(x=list(castmap.keys()), y=list(castmap.values()))

plt.xticks(rotation=90)
casts = df.cast.dropna().str.replace(" ", "")
from wordcloud import WordCloud



text = ""

for word in casts:

  text = text+' '+word



# Generate a word cloud image

wordcloud = WordCloud(background_color="white", max_words=20000).generate(text)



# Display the generated image the matplotlib way

plt.figure(figsize = (16, 9), facecolor = None) 

plt.imshow(wordcloud, interpolation='hermite')

plt.axis("off")

plt.show()
f,ax=plt.subplots(1,2,figsize=(16,9))



sb.countplot('type',data=df,ax=ax[0],order=df['type'].value_counts().index)

df['type'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[1],shadow=True, colors='rgb')



ax[0].set_title('Volume of Movie Types')

ax[1].set_title('Movie Counts per Type')



plt.show()