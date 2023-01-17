import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns
import geopandas as gpd
from wordcloud import WordCloud

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
sns.set()

%matplotlib inline
dataset = pd.read_csv('../input/movies/movies.csv', encoding = "ISO-8859-1")
dataset.head()
sns.heatmap(dataset.isnull(), cbar=False) 
plt.title('Valores faltantes por columna y posición', fontsize = 15)
plt.show()
dataset.describe().T
sns.distplot(dataset['year'], bins = 5, color = 'orange', label = 'KDE')
plt.legend()
plt.gcf().set_size_inches(12, 5)
Oldest = dataset.sort_values("released", ascending = True)
Oldest[['name', "released"]][:10]
Newest = dataset.sort_values("released", ascending = False)
Newest[['name', "released"]][:10]
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) # Loading gpd file
world.head(3)
country_geo = list(world['name']) # Countries in 'naturalearth_lowres' 
country_data = list(dataset['country'].unique()) # Countries in my dataset

country_diff = [country for country in country_data if country not in country_geo]
country_diff # Countries with different names
dataset['country'] = pd.DataFrame(dataset['country'].replace(
    {'USA':'United States of America','UK':'United Kingdom',
     'West Germany':'Germany', 'Hong Kong':'China',
     'Soviet Union': 'Russia', 'Czech Republic':'Czech Rep.'})) # Changing country name from my dataset
Countries = pd.DataFrame(dataset['country'].value_counts())
Ten_countries = pd.DataFrame(dataset['country'].value_counts()).head(10)

sns.barplot(x = Ten_countries.index, y = Ten_countries['country'])

labels =Ten_countries.index.tolist()
plt.gcf().set_size_inches(15, 7)

plt.title('Countries vs movies released', fontsize = 20)
plt.xlabel('Country', fontsize = 15)
plt.ylabel('Movies released', fontsize = 15)

plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9] , labels = labels, rotation = '45')
plt.show()
Temp = Countries.index.to_frame(index=False, name = 'countries')
Temp2 = Countries.reset_index(drop = True)
Temp2 = Temp2.rename(columns={'country': 'Total_movies'})
Temp3 = Temp.join(Temp2)
mapped = world.set_index('name').join(Temp3.set_index('countries')).reset_index()

to_be_mapped = 'Total_movies'
vmin, vmax = 0,4900
fig, ax = plt.subplots(1, figsize=(15,15))

mapped.dropna().plot(column=to_be_mapped, cmap='Blues', linewidth=0.9, ax=ax, edgecolors='0.6')
ax.set_title('Movies per country', fontdict={'fontsize':20})
ax.set_axis_off()

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin = vmin, vmax = vmax))
sm._A = []

cbar = fig.colorbar(sm, orientation='horizontal')
Per_country = (Countries.sum() / 6820 * 100)
Per_country
dataset.groupby('company').size()
company = dataset['company'].value_counts()
company = pd.DataFrame(company) 
company = company.head(10) 
company.head(3)
sns.barplot(x = company.index, y = company['company'])

labels = company.index.tolist()
plt.gcf().set_size_inches(15, 7)

plt.title('Company vs. Movies released', fontsize = 20)
plt.xlabel('Company', fontsize = 15)
plt.ylabel('Released movies', fontsize = 15)
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9] , labels = labels, rotation = '45')
plt.show()
Porcentaje = company.sum() / dataset.shape[0] * 100
Porcentaje
dataset['rating'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(10,8))
plt.title('Rating percentages', fontsize = 20)
plt.tight_layout()
plt.show()
plt.figure(figsize = (22,10))
sns.countplot(x = 'rating',data = dataset ,hue='genre')
plt.legend(loc='upper center')
plt.show()
tag = "Adventure"
small = dataset[dataset["genre"] == tag]
small[small["country"] == "United States of America"][["name", "country","year"]].head(10)
sns.heatmap(dataset.corr(), annot = True, linewidths=.5, cmap='cubehelix')
plt.title('Correlation', fontsize = 20)
plt.gcf().set_size_inches(15, 7)
plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)

plt.gcf().set_size_inches(15, 7)
ax1.scatter(dataset.budget, dataset.gross, c = 'green')
ax1.set_title('Budget vs. Gross', c = 'green', fontsize = 25)
ax2.scatter(dataset.votes, dataset.gross, c='red')
ax2.set_title('Votes vs. Gross', c ='red', fontsize = 25)

plt.ylabel('Gross', fontsize = 25)

plt.show()
plt.subplots(figsize=(12,8))
wordcloud = WordCloud(
                          background_color='Black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(dataset.star))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
plt.subplots(figsize=(12,8))
wordcloud = WordCloud(
                          background_color='White',
                          width=1920,
                          height=1080
                         ).generate(" ".join(dataset.director))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
x1 = dataset['runtime'].fillna(0.0).astype(float)
fig = ff.create_distplot([x1], ['Runtime'], bin_size=0.7, curve_type='normal', colors=["#6ad49b"])
fig.update_layout(title_text='Runtime with normal distribution')
fig.show()
x2 = dataset['score'].fillna(0.0).astype(float)
fig = ff.create_distplot([x2], ['Score'], bin_size=0.1, curve_type='normal', colors=["#6ad49b"])
fig.update_layout(title_text='Score with normal distribution')
fig.show()