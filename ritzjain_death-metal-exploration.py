# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from ipywidgets import interactive, interact

from IPython.display import display

from ipywidgets import widgets

from scipy.ndimage import gaussian_gradient_magnitude

import tabulate

# Any results you write to the current directory are saved as output.



import os



os.listdir('../input/death-metal/')
import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

# import mpld3

# mpld3.enable_notebook()

# import plotly.express as px

sns.set()

mpl.rcParams['figure.figsize'] = (40, 10)
reviews = pd.read_csv('../input/death-metal/reviews.csv')

bands = pd.read_csv('../input/death-metal/bands.csv')

albums = pd.read_csv('../input/death-metal/albums.csv')
reviews.head()
bands.head()
albums.head()
tick_font_dict = {'fontsize' : 15}

label_font_dict = {"fontsize" : 20}

title_font_dict = {'fontsize': 30}
print (f'First time death-metal band formed in: {bands["formed_in"].min()}\nName: {bands.loc[bands.formed_in == 1977, "name"].values[0]}')
print ('Albums created by "Satan\'s host": ')

print('\t' + '\n\t'.join(albums.merge(bands, left_on = 'band', right_on = 'id').query("formed_in == 1977")['title'].values))
satan_album = albums.merge(right = bands, left_on = 'band', right_on = 'id').query('formed_in == 1977')[['id_x', 'title']]

satan_review = satan_album.merge(right = reviews, left_on = 'id_x', right_on = 'album')

group = satan_review.groupby('title_x').agg({"title_x": {"title_count": "count"}})



fig = plt.figure(figsize = (30, 10))

group = group.reset_index()

group.columns = ['_'.join(col) for col in group.columns]

group.columns = ['title_x', 'title_count']

# group.plot(x = 'title_x', y = 'title_count', figsize = (30, 10))

ax = sns.lineplot(x = 'title_x', y = 'title_count', data = group, alpha = 0.8)

sns.scatterplot(x = 'title_x', y = 'title_count', data = group, alpha = 0.5, palette = 'red')

_= ax.set_xticklabels(group['title_x'].values, rotation = 20, fontdict =tick_font_dict)

_ = ax.set_yticklabels(range(0, 7), fontdict= tick_font_dict )

_ = ax.set_xlabel("Album Title", fontdict = label_font_dict)

_ = ax.set_ylabel("Review Counts", fontdict = label_font_dict)

_ = ax.set_title ("Album Review Count for Satan's Host", fontdict = title_font_dict)
ax = bands.groupby('formed_in')['id'].count().plot(rot = 30, marker = 'o', markersize = 5, figsize = (30, 10))

ax.set_xlabel('Years', fontdict = label_font_dict)

ax.set_ylabel('# of bands', fontdict = label_font_dict)

_ = ax.set_xticks(range(1976, 2018))#fontdict

_ = ax.set_xticklabels(range(1976, 2018), fontdict = tick_font_dict)

_ = ax.set_yticks(range(0, 2200, 100))

_ = ax.set_yticklabels(range(0, 2200, 100), fontdict = tick_font_dict)

_ = ax.set_title("Count of bands formed every year", fontdict = title_font_dict)
band_names = bands.loc[:, ['id', 'name']]

album_band_names = albums.merge(band_names, left_on = 'band', right_on = 'id').loc[:, ['id_x', 'name', 'year']]

review_album_name = reviews.merge(album_band_names, left_on = 'album', right_on = 'id_x')

# review_album_name.head()

reviews_overtime = review_album_name.groupby(['year', 'name']).agg({'score': 'sum'}).reset_index()

index = reviews_overtime.groupby('year').agg({'score': 'idxmax'})

filtered_bands = reviews_overtime.loc[index.loc[:,'score'].values]



plt.figure(figsize=(30, 10))

bar = sns.barplot(x = filtered_bands.loc[:, 'year'], y = filtered_bands.loc[:, 'score'])

for p, name in zip(bar.patches, filtered_bands.loc[:, 'name']):

        _x = p.get_x() + p.get_width() - 0.1

        _y = p.get_y() + p.get_height() + 1

        value = name

        bar.text(_x, _y, value, ha="center", rotation = 80, fontdict = {"fontsize": 20})
## Highest bands formed in country

country_wise = bands.groupby('country').agg({'country': {'count'}}).reset_index()

country_wise.columns = ['_'.join(col) for col in country_wise.columns]

top5_country = country_wise.sort_values('country_count', ascending = False).iloc[:10]

print ("Top 10 country of highest bands formation: ")

print ("{:<30} {}".format('Country name', '# of bands'))

for country, count in top5_country.values:

    print ('{:<30} {}'.format(country, count))
## All Death metal bands form in India

plt.figure(figsize = (30, 10))

indian_bands = bands.loc[bands.country == 'India', ['name', 'formed_in', 'country']]

india_band_count = indian_bands.groupby('formed_in').agg({"formed_in": {'count'}}).reset_index()

india_band_count.columns = ['_'.join(col) for col in india_band_count.columns]

ax = sns.barplot(x = 'formed_in_', y = 'formed_in_count', data = india_band_count)

_ = ax.set_xticklabels(india_band_count.formed_in_, rotation = 30)

ax.set_xlabel('Year Formed', fontdict = label_font_dict)

ax.set_ylabel("# of bands", fontdict = label_font_dict)

ax.set_title("Bands formed in India over time", fontdict = title_font_dict)
## Country contribute more in bands

all_country = bands.groupby('country').count()['id'].sort_values()

country_dom = all_country.tail(10)

country_dom['others'] = all_country.sum() - country_dom.sum()

plt.figure(figsize = (10, 10))

_ = plt.pie(x = country_dom.values, labels = country_dom.index,autopct='%1.1f%%', shadow=True)

_= plt.axis('equal')
plt.figure(figsize = (35, 10))

ax = sns.scatterplot(x = 'country_', y = 'country_count', data = country_wise, hue = 'country_', s = 90)

ax.set_xticklabels(country_wise.loc[:, 'country_'], rotation = (90), fontsize = 10, va='top', ha='center',)

ax.set_xlabel('Country Names', fontsize = 20)

ax.set_ylabel("Band Cound", fontsize = 20)

ax.set_title("Band Counts over Country", fontsize = 30)

ax.legend().remove()
## After removing outliers

plt.figure(figsize = (20, 7))

country_wise1 = country_wise[country_wise.country_count < 1000]



ax = sns.scatterplot(x = 'country_', y = 'country_count', data = country_wise1, hue = 'country_', s = 90)

ax.set_xticklabels(country_wise.loc[:, 'country_'], rotation = 90, fontsize = 10)



ax.set_xlabel('Country Names', fontsize = 20)

ax.set_ylabel("Band Cound", fontsize = 20)

ax.set_title("Band Counts over Country", fontsize = 30)

ax.legend().remove()

year_country_group = bands.groupby(['formed_in', 'country']).agg({"country": {'count'}}).reset_index()

year_country_group.columns = [''.join(col) for col in year_country_group.columns]
def show_table(year, country):

    

    if (year == 1975):

        figure = plt.figure(figsize = (15, 6))

        data = year_country_group.query('country == @country')

        ax = sns.scatterplot(x = 'formed_in', y = 'countrycount', data = data, marker = 'o', s = 120, hue = 'formed_in')

        _ = ax.set_xlabel("Years ", fontdict = label_font_dict)

        _ = ax.set_title(f"Bands formed in {country}", fontdict = title_font_dict)

        ax.legend().remove()

        return ax

    else:

        data = year_country_group.query('formed_in == @year and country == @country')

        if (data.shape == (0, 3)):

            display(f'Country {country} has no bands formed in year {year}')

        else:

            print(f'Country {country} has {data.iloc[0].countrycount} bands formed in year {year}')

            print('Name of Bands: ')

            band_names = bands.query('formed_in == @year and country == @country').name.values

            print (tabulate.tabulate(band_names[:, np.newaxis], tablefmt = 'fancy_grid', headers = ['Band\'s Name']))
year = widgets.Dropdown(

    options = list(range(1975, 2017)), 

    value = 1975, 

    description = 'Year: ', 

    disabled = False

)



country = widgets.Dropdown(

    options = sorted(bands.country.unique()), 

    description = 'Country: ',

    value = 'United States'

)

w = interactive(show_table, year = year, country = country)

w
albums.head()
album_year = albums.groupby('year').agg({"year": {"count"}}).reset_index()

album_year.columns = [''.join(col) for col in album_year.columns]

band_year = bands.groupby("formed_in").agg({"formed_in": {"count"}}).reset_index()

band_year.columns = [''.join(col) for col in band_year.columns]
plt.figure(figsize = (10, 6))

plt.plot(band_year.formed_in, band_year.formed_incount, '-o')

plt.plot(album_year.year, album_year.yearcount, '-^')

plt.xlabel("Year")

plt.title("Album and Band Trade off")

plt.legend(['Bands', 'Albums'])
band_count = bands.groupby('formed_in')['id'].count().cumsum()

album_count = albums.groupby('year')['id'].count().cumsum()



fig = plt.figure(figsize = (10, 6))

# plt.plot(band_count.index, band_count, '-o', )

# plt.plot(album_count, '-o')

# dir(fig)



band_count.plot(marker = 'o')

ax = album_count.plot(marker = 'o')

ax.set_xlabel("Year", fontdict = label_font_dict)

ax.legend(['# of Band', '# of Album'], loc = 'best')

_ = ax.set_title("Album and Bands over years", fontdict = title_font_dict)
country = widgets.Dropdown(options = sorted(bands.country.unique()), 

                          value = 'India',

                          description= "Country: ")



@interact(country = country)

def show_country_plot(country):

    band_data = bands.query('country == @country').loc[:, ['id', 'formed_in', 'country']]

    album_data = albums.loc[albums.band.isin(band_data.id), ['id', 'year']]

    plt.figure(figsize = (10, 5))

    band_count = band_data.groupby('formed_in')['id'].count().cumsum()

    album_count = album_data.groupby('year')['id'].count().cumsum()

    ax = band_count.plot(marker = 'o', markersize = 5, alpha = 0.8)

    album_count.plot(marker = 'o', markersize = 5, alpha = 0.8)

    ax.legend(['Band Count', 'Album Count'], loc = 'best')
# highest number of albums released by country



band_id = bands.loc[:, ['country', 'id']]

# albums.merge(band_id, left_on = 'band', right_on='id', how  = 'inner')

country_album = pd.merge(left = band_id, right = albums, how = 'inner', left_on = 'id', right_on = 'band').loc[:, ['country', 'year', 'id']]

country_album = country_album.groupby(['year', 'country']).size().reset_index()

index = country_album.groupby(['year']).agg({0: np.argmax}).reset_index().loc[:, 0]

country_album = country_album.iloc[index]
plt.figure(figsize = (15, 6))

ax = sns.barplot(x = country_album.year, y = country_album.loc[:, 0])

ax.set_xticklabels(country_album.year, rotation = 45)

ax.set_xlabel("Year", fontdict = label_font_dict)

ax.set_ylabel("Album released by country", fontdict = label_font_dict)

ax.set_title("Max Albums released by country", fontdict = title_font_dict)

for p, name in zip(ax.patches, country_album.country):

        _x = p.get_x() + p.get_width() - 0.1

        _y = p.get_y() + p.get_height() + 1

        value = name

        ax.text(_x, _y, value, ha="center", rotation = 80, fontdict = {"fontsize": 10})
album_review = reviews.groupby(['album']).agg({"score": {"sum"}}).reset_index().sort_values(('score', 'sum'))

album_review = album_review.merge(albums, left_on = 'album', right_on = 'id').loc[:, ['title', ('score', 'sum')]].reset_index().tail(10)



ax = sns.barplot(x = ('score', 'sum'), y = 'title', data = album_review, )

ax.set_xlabel('Score')

ax.set_ylabel("Album Name")

ax.set_title('Top 10 Most Liked Albums')
import re

remove_words = re.compile(r'\b(\w*later\w*)\b|\b(\w*early\w*)\b|\b(\w*present\w*)\b|\b(\w*metal\w*)\b|\b(\w*with\w*)\b')

remove_punct = re.compile(r'[\(\)\d\';&^%$#@!"-]')

remove_mul_space = re.compile(r'\s+')
genre = bands.genre.str.lower().map(lambda x: remove_words.sub('', x))

genre = genre.map(lambda x: remove_punct.sub(' ', x))

genre_df = genre.str.split(r'[|/]', expand = True)

genre = pd.concat([genre_df.loc[:, col] for col in genre_df.columns])

genre = genre.str.strip().apply(lambda x: None if (x == '') else x)

genre = genre.dropna(axis = 0)

genre = genre.apply(lambda x: remove_mul_space.sub(' ', x))
print (f"Total unique genre in Death Metal Genre: {genre.unique().shape[0]}")
print ("List of all Death Metal Genre: ")

print('\n'.join(genre.unique()))
genre_df = pd.DataFrame(genre, columns = ['genre'])

genre_count = genre_df.groupby('genre').agg({'genre': {"count": 'count'}}).reset_index()

genre_count.columns = ['_'.join(col) for col in genre_count.columns]
genre = bands.genre.map(lambda x: remove_punct.sub(' ', x))

genre_df = genre.str.split(r'[|/]', expand = True)

genre = pd.concat([genre_df.loc[:, col] for col in genre_df.columns])

genre = genre.str.strip().apply(lambda x: None if (x == '') else x)

genre = genre.dropna(axis = 0)

genre = genre.apply(lambda x: remove_mul_space.sub(' ', x))
import requests

from io import BytesIO

from PIL import Image

urlq = 'https://free4kwallpapers.com/uploads/originals/2018/06/14/colorful-skull-wallpaper.jpg'

red = 'https://free4kwallpapers.com/uploads/originals/2018/03/14/red-skull-ascii-wallpaper.jpg'

req = requests.get(urlq)

img = Image.open(BytesIO(req.content))
img_array = np.array(img)

img_array = img_array[::3, ::3]

img_mask = img_array.copy()

img_mask[img_mask.sum(axis=2) == 0] = 255

edges = np.mean([gaussian_gradient_magnitude(img_array[:, :, i] / 255., 2) for i in range(3)], axis=0)

img_mask[edges > .08] = 0
import wordcloud as wc



plt.figure(figsize = (20, 10))



cloud = wc.WordCloud(width = 1280, height = 720, max_words = 1000, mask = img_mask, background_color='black').generate(' '.join(genre.values))

img_color = wc.ImageColorGenerator(img_mask)

cloud.recolor(color_func = img_color)

plt.grid(False)

plt.imshow(cloud, interpolation='bilinear')
genre_data = bands.groupby("genre").count()['id'].sort_values()



genre_val = genre_data.tail(10)

genre_val['others'] = genre_data.sum() - genre_val.sum()

genre_val
plt.figure(figsize=  (10, 5))

plt.pie(x = genre_val.values, labels = genre_val.index);