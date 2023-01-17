import numpy as np
np.warnings.filterwarnings('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
%config InlineBackend.figure_format='retina'
import seaborn as sns

import pandas as pd
import re
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
df = pd.read_csv('../input/22000-scotch-whisky-reviews/scotch_review.csv', index_col = 0)
df.index = df.index - 1   #remove 1 so that the index starts from 0

print('This dataset has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))
df.head()
df.rename(columns = {'review.point': 'points'}, inplace = True)
df.columns
df.info()
symbol_idx = pd.to_numeric(df['price'], errors = 'coerce').isnull() # errors = 'coerce' results in NaNs for non-numeric values
df[symbol_idx][['name','price']].head()
df.at[[19, 95, 410, 1000, 1215], 'price'] = 15000   # instances with '60,000/set' which equals 15000 dollars
df['price'].replace('/liter', '', inplace = True, regex = True) # this bottle was actually 1 lt, so we don't need the price per litre
df['price'].replace(',', '', inplace = True, regex = True)

df['price'] = df['price'].astype('float')
df['currency'].value_counts()
df.drop('currency', axis = 1, inplace = True)
df['price_p_points'] = df['price']/df['points']
df.head()
df['age'] = df['name'].str.extract(r'(\d+) year')[0].astype(float) # extract age and convert to float

df['name'] = df['name'].str.replace(' ABV ', '')
df['alcohol%'] = df['name'].str.extract(r"([\(\,\,\'\"\’\”\$] ? ?\d+(\.\d+)?%)")[0]
df['alcohol%'] = df['alcohol%'].str.replace("[^\d\.]", "").astype(float) # keep only numerics and convert to float

df[['name', 'age', 'alcohol%']].sample(10, random_state = 42)
df[['age', 'alcohol%']].isnull().sum()
df.describe()
attributes = ['price', 'points', 'age', 'alcohol%']  # price_p_points not that important

df[attributes].hist(figsize = (15, 10), color = 'firebrick');
df['category'].value_counts()
colors = ['#BB342F', '#EDAFB8', '#666A86', '#95B8D1', '#D1D0A3']
categories_index = df['category'].value_counts().index

fig = plt.figure(figsize = (7, 4))
sns.countplot(y = 'category', data = df, palette = colors, order = df['category'].value_counts().index)

# include the percentage of each category next to each bar
for index, value in enumerate(df['category'].value_counts()):
    label =  '{}%'.format(round( (value/df['category'].shape[0])*100, 2)) 
    plt.annotate(label, xy = (value + 11, index + 0.1), color = colors[index])

plt.title('Number of bottles by category (with percentages)')
plt.ylabel('Category')
plt.xlabel('Count');
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (13, 5))

ax1.set_title('Mean price by category')
sns.barplot(x = 'price', y = 'category', data = df, order = categories_index, palette = colors, ax = ax1)

ax2.set_title('Mean points by category')
sns.barplot(x = 'points', y = 'category', data = df, order = categories_index, palette = colors, ax = ax2)
ax2.set(yticklabels = [])
ax2.set_ylabel('')
ax2.set_xlim(80, 89);
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 6))

sns.boxplot(x = 'price', y = 'category', data = df, order = categories_index, palette = colors, showfliers = False, ax = ax1)
sns.boxplot(x = 'points', y = 'category', data = df, order = categories_index, palette = colors, ax = ax2)
ax2.set(yticklabels = [])
ax2.set_ylabel('')

plt.tight_layout();
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (13, 5))

ax1.set_title('Mean age by category')
sns.barplot(x = 'age', y = 'category', data = df, order = categories_index, palette = colors, ax = ax1)

ax2.set_title('Mean alcohol% by category')
sns.barplot(x = 'alcohol%', y = 'category', data = df, order = categories_index, palette = colors, ax = ax2)
ax2.set(yticklabels = [])
ax2.set_ylabel('')
ax2.set_xlim(30, 55);
from wordcloud import WordCloud, STOPWORDS

def create_word_cloud(df, bg_color, max_words, mask, stop_words, max_font_size, colormap):
    
    wc = WordCloud(background_color = bg_color, max_words = max_words, mask = mask, stopwords = stop_words, max_font_size = max_font_size)
    wc.generate(' '.join(df))
    
    return wc.recolor(colormap = colormap, random_state = 42)
sm_df = df[df['category'] == 'Single Malt Scotch']['description'].values
bd_df = df[df['category'] == 'Blended Scotch Whisky']['description'].values
from PIL import Image
import requests # https://stackoverflow.com/questions/12020657/how-do-i-open-an-image-from-the-internet-in-pil/12020860

mask_sm = np.array(Image.open(requests.get('https://imgur.com/dAYKIVT.png', stream = True).raw))  # mask for single malts
mask_bd = np.array(Image.open(requests.get('https://imgur.com/upL1TBW.png', stream = True).raw))  # mask for blended whiskies
stop_words = list(STOPWORDS)

fig, ax = plt.subplots(2, 1, figsize = (7, 14))

ax[0].imshow(create_word_cloud(df = sm_df, bg_color = 'white', max_words = 50, mask = mask_sm, stop_words = stop_words,
                             max_font_size = 50, colormap = 'winter'), alpha = 1, interpolation = 'bilinear')
ax[0].set_title('Single Malts - Initial', size = 16, y = 1.06)
ax[0].axis('off')

ax[1].imshow(create_word_cloud(df = bd_df, bg_color = 'white', max_words = 50, mask = mask_bd, stop_words = stop_words,
                             max_font_size = 50, colormap = 'tab10'), alpha = 1, interpolation = 'bilinear')
ax[1].set_title('Blended Whiskies - Initial', size = 16, y = 1.04)
ax[1].axis('off');
stop_words = ['whisky', 'whiskies', 'blend', 'note', 'notes', 'year', 'years', 'old', 'nose', 'finish', 'bottle',
              'bottles', 'bottled', 'along', 'release', 'flavor', 'cask', 'well', 'make', 'mouth', 'palate', 'hint',
              'one', 'bottling', 'distillery', 'quite', 'time', 'date', 'show', 'first'] + list(STOPWORDS)
fig, ax = plt.subplots(2, 1, figsize = (7, 14))

ax[0].imshow(create_word_cloud(df = sm_df, bg_color = 'white', max_words = 50, mask = mask_sm, stop_words = stop_words,
                             max_font_size = 50, colormap = 'winter'), alpha = 1, interpolation = 'bilinear')
ax[0].set_title('Single Malts - Improved', size = 16, y = 1.06)
ax[0].axis('off')

ax[1].imshow(create_word_cloud(df = bd_df, bg_color = 'white', max_words = 50, mask = mask_bd, stop_words = stop_words,
                             max_font_size = 50, colormap = 'tab10'), alpha = 1, interpolation = 'bilinear')
ax[1].set_title('Blended Whiskies - Improved', size = 16, y = 1.04)
ax[1].axis('off');
df.sort_values(by = 'price', ascending = False).head()
df.sort_values(by = 'price', ascending = True).head()
df.sort_values(by = 'points', ascending = False).head()
df.sort_values(by = 'price_p_points', ascending = True).head()
df[(df['points'] > 85) & (df['price'] < 50)].sort_values(by = 'points', ascending = False).head()
df.sort_values(by = 'age', ascending = False).head()
df.sort_values(by = 'alcohol%', ascending = False).head()
