import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from PIL import Image

from wordcloud import WordCloud, STOPWORDS

import requests

from io import BytesIO
wineDFRaw = pd.read_csv('/kaggle/input/wine-reviews/winemag-data-130k-v2.csv', index_col=0)

wineDF = wineDFRaw[['country','points', 'price', 'province', 'variety', 'winery', 'taster_name', 'description']]
# Swapping color for masking from black to white

def transform_format(val):

    if val == 0:

        return 255

    else:

        return val
text = " ".join(review for review in wineDF.description)

stopwords = set(STOPWORDS)

stopwords.update(["drink", "now", "wine", "flavor", "flavors"])



# Get image for the mask

wineMask = np.array(Image.open("../input/wine-mask/wine_mask.png"))



# Transforming wineMask dengan fungsi transform_format

transformedWineMask = np.ndarray((wineMask.shape[0],wineMask.shape[1]), np.int32)

for i in range(len(wineMask)):

    transformedWineMask[i] = list(map(transform_format, wineMask[i]))



wc = WordCloud(background_color="white", max_words=1000, mask=transformedWineMask, stopwords=stopwords, contour_width=3, contour_color='firebrick').generate(text)

plt.figure(figsize=[20,10])

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
wineDF.head(10)
plt.style.use('fivethirtyeight')

g = sns.catplot(x='country', data=wineDF, kind='count', order=pd.value_counts(wineDF['country']).iloc[:10].index, palette= 'ocean')

g.ax.set_title('Top 10 Country Dengan Produksi Wine Terbanyak')

g.set(xlabel= 'Country', ylabel='Jumlah Produksi Wine')

g.fig.set_size_inches([12,7])

g.set_xticklabels(rotation=45)

for p in g.ax.patches:

    g.ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.13, p.get_height()+500))

plt.tight_layout()
df = wineDF.loc[:, ['country', 'points']]

df = df.groupby('country')['points'].mean().head(10)

df = df.sort_values(ascending=False)

g = sns.barplot(x=df,y= df.index, palette= 'plasma')

g.set_title('Top 10 Country Dengan Rata-Rata Rating Wine Tertinggi')

g.set(xlabel='Rating Wine', ylabel='Country')

g.set_xlim([50,100])

plt.gcf().set_size_inches(15,5)

for p in g.patches:

    width = p.get_width()

    plt.text(2+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:1.2f}'.format(width),

             ha='center', va='center')

plt.show()
wineDF['price'].quantile([0, 0.25, .75, .9])
def price_category(price):

    if price <= 17.0:

        return 'Murah'

    elif price > 17 and price <= 42:

        return 'Sedang'

    else:

        return 'Mahal'
wineDF['price_category'] = wineDF['price'].apply(price_category)
g = sns.catplot(x='country', data=wineDF, kind='count', order=pd.value_counts(wineDF['country']).iloc[:10].index, hue='price_category', hue_order=['Murah', 'Sedang', 'Mahal'], palette= 'inferno')

g.ax.set_title('Kategori Harga Wine Setiap Negara Dengan Produksi Wine Terbanyak')

g.set(xlabel= 'Country', ylabel='Jumlah Wine')

g.fig.set_size_inches([20,7])

g.set_xticklabels(rotation=45)

for p in g.ax.patches:

    g.ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x(), p.get_height()+500))

plt.tight_layout()
g = sns.catplot(y='variety', data=wineDF, kind='count', order=pd.value_counts(wineDF['variety']).iloc[:10].index, palette= 'winter')

g.ax.set_title('Top 10 Jenis Anggur yang Digunakan')

g.set(xlabel= 'Jenis Buah Anggur', ylabel='Jumlah Produksi Wine')

g.fig.set_size_inches([20,7])

for p in g.ax.patches:

    width = p.get_width()

    plt.text(400+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

plt.tight_layout()
wineDF['points'].quantile([0, 0.25, .75, .9])
def rating_category(point):

    if point <= 86.0:

        return 'Tidak Enak'

    elif point > 86.0 and point <= 91.0:

        return 'Lumayan'

    else:

        return 'Enak'
wineDF['rating_category'] = wineDF['points'].apply(rating_category)
g = sns.catplot(y='variety', data=wineDF, kind='count', order=pd.value_counts(wineDF['variety']).iloc[:10].index, hue='rating_category', hue_order=['Tidak Enak', 'Lumayan', 'Enak'], palette= 'cool', height=6, aspect=2)

g.ax.set_title('Kategori Rating Wine Setiap Negara Dengan ')

g.set(xlabel= 'Jumlah Wine', ylabel='Jenis Anggur')

g.fig.set_size_inches([20,10])

g.ax.set_xlim([0,7000])

g.ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000])

for p in g.ax.patches:

    width = p.get_width()

    plt.text(250+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

plt.show()
wineDFClean = wineDF[wineDF['country'] == 'US']

wineDFClean.head(10)
df = wineDFClean.loc[:, ['winery', 'points']]

df.groupby('winery')['points'].mean()

df = df.sort_values('points', ascending=False)

df = df.head(20)

g = sns.catplot(x='points', y='winery',data= df, kind='bar', palette= 'spring')

g.ax.set_xlim([80,100])

g.set(xlabel='Rating', ylabel='Winery')

g.ax.set_title('Winery dengan Rating Rata-Rata Tertinggi di US')

g.fig.set_size_inches([15,7])

plt.tight_layout()
df = wineDFClean.loc[:, ['variety', 'points']]

df.groupby('variety')['points'].mean()

df = df.sort_values('points', ascending=False)

df = df.head(150)

g = sns.catplot(x='points', y='variety',data= df, kind='bar', palette='autumn')

g.ax.set_xlim([80,100])

g.set(xlabel='Rating', ylabel='Tipe Anggur')

g.ax.set_title('Tipe Buah Anggur dengan Rating Rata-Rata Tertinggi di US')

g.fig.set_size_inches([15,7])

plt.tight_layout()
df = wineDFClean.loc[:, ['price', 'points', 'price_category']]

g = sns.relplot(x='points', y='price',data= df, kind='scatter', color='g')

g.set(xlabel='Rating', ylabel='Harga')

g.ax.set_title('Perbandingan Rating dengan Harga Jual Wine di US')

g.fig.set_size_inches([10,5])

plt.tight_layout()
df = df.sort_values('points')

g = sns.catplot(x='points', y='price_category', data=df, kind='bar', palette='Blues')

g.set(xlabel='Rating', ylabel='Kategori Harga')

g.ax.set_title('Perbandingan Rating dengan Harga Jual Wine')

g.fig.set_size_inches([15,5])

g.ax.set_xlim([50,100])

# Annotate

for p in g.ax.patches:

    width = p.get_width()

    plt.text(3+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:1.2f}'.format(width),

             ha='center', va='center')

plt.tight_layout()