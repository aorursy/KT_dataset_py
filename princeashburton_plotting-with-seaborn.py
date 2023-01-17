# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
import seaborn as sns
sns.countplot(reviews['points'])
sns.kdeplot(reviews.query('price < 200').price)
reviews[reviews['price'] < 200]['price'].value_counts().sort_index().plot.line()
sns.kdeplot(reviews[reviews['price'] < 200].loc[:,['price', 'points']].dropna().sample(5000))
sns.distplot(reviews['points'], bins=10,kde=False)
sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100])
#Hexplot
sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100], kind='hex',gridsize=20)
df = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]

sns.boxplot(
     x='variety',
      y='points',
     data=df)
sns.violinplot(x='variety',y='points',
              data =reviews[reviews.variety.isin(reviews.variety.value_counts()[:5].index)]
              )
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head()
sns.countplot(pokemon['Generation'])
sns.distplot(pokemon['HP'])
sns.jointplot(x='Attack',y='Defense',data=pokemon)
sns.jointplot(x='Attack',y='Defense',data=pokemon, kind='hex', gridsize=20)
sns.kdeplot(pokemon['HP'], pokemon['Attack'] )
sns.boxplot(x='Legendary',y='Attack',data=pokemon)
sns.violinplot(x='Legendary',y='Attack',data=pokemon)