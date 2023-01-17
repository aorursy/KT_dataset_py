import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
cartier = pd.read_csv('../input/cartier-jewelry-catalog/cartier_catalog.csv')
cartier.head()
cartier.info()
# There is no sence in reference number from analysis point of view, so I'm removing this column
cartier.drop('ref', axis = 1, inplace = True)

# Also I won't analyse image in this notebook
cartier.drop('image', axis = 1, inplace = True)
# Checking uniques values in categories
cartier['categorie'].unique()
cartier.rename(columns = {'categorie': 'category'}, inplace = True)
cartier['category'] = cartier['category'].astype('category')
# Checking title column
cartier['title'].unique()
cartier.drop('title', axis = 1, inplace = True)
# Checking price distribution
cartier['price'].describe()
# Checking the most expensive items
cartier[cartier['price'] > 300000]
# Checking uniques tags values
set([tag for tags in cartier['tags'].str.replace('.','').str.split(', ') for tag in tags])
# Creating list with metals, coatings and crystals based on set of unique tags
metals = ['yellow gold', 'platinum', 'pink gold', 'white gold', 'non-rhodiumized white gold']
coatings = ['black lacquer', 'lacquer', 'black ceramic', 'ceramic']
crystals = ['amazonite', 'amethyst', 'amethysts', 'aquamarines', 'aventurine', 'brown diamonds', 'carnelian', 'carnelians', 'chrysoprase', 'chrysoprases', 'citrine', 'coral', 'diamond', 'diamonds', 'emeralds', 'garnets', 'gray mother-of-pearl', 'lapis lazuli', 'malachite', 'mother-of-pearl', 'obsidians', 'onyx', 'orange diamonds', 'pearl', 'peridots', 'pink sapphire', 'pink sapphires', 'rubies', 'sapphire', 'sapphires', 'spessartite garnet', 'spinels', 'tsavorite garnet', 'tsavorite garnets', 'white mother-of-pearl', 'yellow diamonds']

# Initialising functions to divide tags in different categories. 
def check_tags(group, tags):
    value = ''
    for tag in tags:
        if tag in group:
            value += tag.rstrip('s') + ', '
    if value == '':
        return 'No'
    return value.rstrip(", ")
    
def metal(tags):
    return check_tags(metals, tags)
def crystal(tags):
    return check_tags(crystals, tags)
def coating(tags):
    return check_tags(coatings, tags)
# Creating new columns with metals, crystals and coatings instead of tags 
cartier['metals'] = cartier['tags'].str.replace('non-rhodiumized white gold','white gold').str.replace('.','').str.split(', ').apply(metal)
cartier['crystals'] = cartier['tags'].str.replace('rubies','ruby').str.replace('.','').str.split(', ').apply(crystal)
cartier['coatings'] = cartier['tags'].str.replace('.','').str.split(', ').apply(coating)
# Removing tags in a separate column
tags = cartier.pop('tags')
# Checking descriptions' lenth
cartier['description'].str.len().sort_values(ascending = False)
cartier.iloc[258,2]
cartier.iloc[691,2]
cartier.iloc[258,2] = 'Clash de Cartier ring, XL model, 18K yellow gold, coral. Width: 17.7mm.'
cartier.iloc[691,2] = 'Clash de Cartier earrings, XL model, 18K yellow gold, coral. Width: 17.7mm.'
cartier['title'] = cartier['description'].apply(lambda x: x.split(', ')[0].split(' - ')[0])
cartier['width'] = cartier['description'].apply(lambda x: x.split('Width: ')[1].split('mm')[0] if len(x.split('Width: ')) > 1 else np.nan).astype('float')
# Removing old column
description = cartier.pop('description')
cartier.info()
plt.figure(figsize = (10, 6))
price = sns.distplot(cartier['price'], kde = False, color="r", bins = 50)
price.set_xlabel('Price')
cartier['price'].describe()
plt.figure(figsize = (10, 6))
price_category = sns.swarmplot(y = 'category', x = 'price', data = cartier, palette = 'magma')
price_category.set_xlabel('Price')
price_category.set_ylabel('Category')
plt.figure(figsize = (10, 8))
price_metal = sns.swarmplot(y = 'metals', x = 'price', data = cartier, palette = 'magma_r')
price_metal.set_xlabel('Price')
price_metal.set_ylabel('Metal')
plt.figure(figsize = (10, 6))
all_metals = [metal for rows in cartier['metals'].str.split(', ') for metal in rows]
metals = sns.countplot(all_metals, palette = 'magma_r')
metals.set_xlabel('Metal')
metals.set_ylabel('Jewelry count')
all_coatings = [metal for rows in cartier[cartier['coatings'] != 'No']['coatings'].str.split(', ') for metal in rows]
coatings = sns.countplot(all_coatings, palette = 'magma_r')
coatings.set_xticklabels(coatings.get_xticklabels(), rotation=90)
coatings.set_xlabel('Coating')
coatings.set_ylabel('Jewelry count')
plt.figure(figsize = (10, 8))
price_metal = sns.swarmplot(y = 'coatings', x = 'price', data = cartier, palette = 'magma_r')
price_metal.set_xlabel('Price')
price_metal.set_ylabel('Coating')
plt.figure(figsize = (12, 8))
all_crystals = [metal for rows in cartier[cartier['crystals'] != 'No']['crystals'].str.split(', ') for metal in rows]
crystals = sns.countplot(y = all_crystals, palette = 'Blues_r', order = pd.Series(all_crystals).value_counts().index, )
crystals.set_xticklabels(crystals.get_xticklabels(), rotation=90)
crystals.set_xlabel('Jewelry count')
crystals.set_ylabel('Crystals')
from wordcloud import WordCloud, STOPWORDS

# Setting stopwords
stopwords_names = set(STOPWORDS)
stopwords_names.update(['ring', 'Ring', 'bracelet', 'Bracelet', 'necklace', 'Necklace', 'earrings', 'Earrings', 'de', 'Cartier'])

# Creating words list for names
words_in_title = [word for rows in cartier['title'].str.split() for word in rows if word not in stopwords_names]
words = " ".join(word for word in words_in_title)
# Creating a cloud with words from names:
plt.figure(figsize = (10,6))
wordcloud = WordCloud(max_words=30, background_color="white", colormap = 'tab20b').generate(words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()