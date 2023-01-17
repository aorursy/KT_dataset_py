import pandas as pd

restaurants = pd.read_csv("../input/yellowpages_com-restaurant_sample (1).csv")
restaurants.head(3)
import numpy as np

import itertools



def catmap(cats):

    if pd.isnull(cats):

        return [np.nan]

    else:

        return [cat.strip() for cat in cats.split(",")]



# Generate the by-entry categories list.

cat_lists = restaurants['Categories'].map(catmap)



# Get the set of possible categories.

from itertools import chain

categories = set(list(chain.from_iterable(cat_lists.values.tolist())))
len(categories)
from tqdm import tqdm



cat_lists = restaurants['Categories'].map(catmap)



for cat in tqdm(categories):

    restaurants[cat] = cat_lists.map(lambda cats: cat in cats)
restaurants.head(3).shape
category_counts = restaurants.loc[:, categories].sum().sort_values(ascending=False)

category_counts = category_counts.drop('Restaurants')  # every entry is a restaurant
import seaborn as sns
import matplotlib.pyplot as plt

category_counts.head(20).plot.bar(title='Top 20 Most Common Restaurant Food Categories',

                                  figsize=(14, 7))

plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45, ha='right', fontsize=14)

pass
restaurants.loc[:, categories - {'Restaurant'}].T.sum().value_counts().sort_index().plot.bar(

    title='Restaurants by Number of Yellowpages Categories Assigned', figsize=(14, 7),

    fontsize=16

)
"Restaurants are in {:.2f} categories on average".format(

    restaurants.loc[:, categories - {'Restaurant'}].T.sum().sum() / len(restaurants)

)
import missingno as msno

msno.matrix(restaurants.loc[:, categories - {'Restaurants'}].replace(False, np.nan).head(500))
display_categories = set(category_counts[category_counts > 50].index) - {'Restaurants'}
msno.dendrogram(

    restaurants.loc[:, display_categories].replace(False, np.nan),

    orientation='left',

    figsize=(9, 14)

)
import seaborn as sns

restaurants['State'].value_counts().plot.bar(title='Restaurants by State', figsize=(14, 7), 

                                             fontsize=14)
indiana_tot = restaurants.query('State == "IN"').loc[:, categories - {'Restaurants'}].sum() / len(restaurants.query('State == "IN"'))

florida_tot = restaurants.query('State == "FL"').loc[:, categories - {'Restaurants'}].sum() / len(restaurants.query('State == "FL"'))

penn_tot = restaurants.query('State == "PA"').loc[:, categories - {'Restaurants'}].sum() / len(restaurants.query('State == "PA"'))
(indiana_tot - florida_tot)[

    # Index by restaurant types with the largest difference per capita

    (indiana_tot - florida_tot).abs().sort_values(ascending=False).head(5).index

].sort_values().plot.bar(

    title='Restaurants More Popular in Indiana than Florida, Largest Differences', 

    figsize=(14, 7), fontsize=14

)
(penn_tot - indiana_tot)[

    # Index by restaurant types with the largest difference per capita

    (penn_tot - indiana_tot).abs().sort_values(ascending=False).head(5).index

].sort_values().plot.bar(

    title='Restaurants More Popular in Florida than Pennsylvania, Largest Differences', 

    figsize=(14, 7), fontsize=14

)
restaurants['Email'][restaurants['Email'].notnull()].head()
(restaurants['Email'][restaurants['Email'].notnull()]

     .str.split(".")

     .map(lambda d: d[-1])

     .str.lower()

     .value_counts()

     .plot.bar(figsize=(14, 7), fontsize=14, title='Domain Name Endings Used by Restaurants'))
(restaurants['Email'][restaurants['Email'].notnull()]

     .str.split("@")

     .map(lambda d: d[0])

     .str.lower()

     .value_counts()

     .head(10)

     .plot.bar(figsize=(14, 7), fontsize=14, title='Domain Name Endings Used by Restaurants'))