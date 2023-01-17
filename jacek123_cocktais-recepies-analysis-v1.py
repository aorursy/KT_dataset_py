# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('/kaggle/input/cocktails-hotaling-co/hotaling_cocktails - Cocktails.csv')
sns.set(style="darkgrid")

def plotDist(data, prop, minimum = 0, title=''):

    plt.figure(figsize=(10,6))

    singnificant_values = data.groupby(prop).filter(lambda x: len(x) >= minimum)

    plot = sns.countplot(

        data = singnificant_values,

        y=prop,

        order=singnificant_values[prop].value_counts().index,

        palette="deep"

    )

    plot.set_title(title)

    plt.tight_layout()

    plt.show()

    return singnificant_values
df.head()
locations = plotDist(df, 'Location', 2, 'Distribution of recepies origin location')
companies = plotDist(df, 'Bar/Company', 3, 'Most innovative Bars/Companies')
companiesLocation = plotDist(companies, 'Location', 0, 'Location of top bars/companies')
top10 = companies['Location'].value_counts()[0:10]

sum(top10) / df.shape[0]
bartenders = plotDist(df, 'Bartender', 4, 'Top Bartenders')
top5Bartenders = bartenders['Bartender'].value_counts()[0:5]



top5BartendersData = bartenders.loc[bartenders['Bartender'].isin(top5Bartenders.index)][['Bartender','Bar/Company']]



plt.figure(figsize=(20,20))

catplot = sns.catplot(y="Bartender", hue="Bar/Company", kind="count", height=6,

            palette="deep", edgecolor=".6",

            data=top5BartendersData,

            legend_out=False)

plt.legend(loc='lower right')

plt.title("Bars/Companies of top Bartenders")

plt.tight_layout()

plt.show()
sum(top5Bartenders) / df.shape[0]
g = df.set_index('Cocktail Name').Garnish.str.split(',', expand=True).stack().reset_index('Cocktail Name').reset_index(drop=True)

g.columns = ['Cocktail Name', 'garnish']

g['garnish'] = g['garnish'].str.replace('[0-9] ', '') #remove numbers

g['garnish'] = g['garnish'].str.replace('rries$','rry') #unify plular and singular

g['garnish'] = g['garnish'].str.replace('Luxardo Cherry', 'Cherry Luxardo') #remove numbers

g['garnish']= g['garnish'].map(str.strip) 

g['garnish_ingr'] = g['garnish'].str.replace('[a-zA-Z]+$','') #split

g['garnish_ingr'] = g['garnish_ingr'][g['garnish_ingr'] != '']

g['garnish_type'] = g['garnish'].str.extract('([a-zA-Z]+)$')

g['garnish_type'] = g['garnish_type'][g['garnish_type'] != 'Luxardo']

g['garnish_type'] = g['garnish_type'].str.capitalize()
garnish = plotDist(g, 'garnish', 5, 'Most popular garnish')
garnish_ingr = plotDist(g, 'garnish_ingr', 4, 'Top garnish ingredients')
garnish_type = plotDist(g, 'garnish_type', 15, 'Garnish method')
i = df.set_index('Cocktail Name').Ingredients.str.split(',', expand=True).stack().reset_index('Cocktail Name').reset_index(drop=True)



i.columns = ['Cocktail Name', 'ingredients']

i['ingredients'] = i['ingredients'].str.replace('.?[0-9]\.?[0-9]?( oz )?', '') #remove quantity

i['ingredients'] = i['ingredients'].str.replace('dash|ml|tsp', '') #remove quantity

i['ingredients'] = i['ingredients'].map(str.strip)

i['ingredients'] = i['ingredients'].str.lower().str.capitalize()

i['ingredients'] = i['ingredients'][i['ingredients'].str.contains('juice') == False]

i['ingredients'] = i['ingredients'][i['ingredients'].str.contains('bitter') == False]
ingredients = plotDist(i, 'ingredients', 20, 'Base Liqueur')
topBartender = top5Bartenders[[0]]



topBartenderData = bartenders.loc[bartenders['Bartender'].isin(topBartender.index)][['Bartender','Cocktail Name']]

j = pd.merge(topBartenderData, i, on='Cocktail Name')



ingredients = plotDist(j, 'ingredients', 3, 'Ingrediends used by Francesco Lafranconi')