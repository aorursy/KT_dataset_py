# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importação e leitura do conjunto de dados

wine_filepath = '../input/wine-reviews/winemag-data-130k-v2.csv'

wine_data = pd.read_csv(wine_filepath, index_col = 0).rename_axis("wines", axis='columns')

wine_data = wine_data.rename(columns={'points': 'score'})





# Tamanho do conjunto de dados

wine_data.shape

# Visualização rápida do conjunto de dados

wine_data.head()
# Setting the NaN countries to Unknown

wine_data.country.fillna('Unknown',inplace=True)



# Setting the NaN taster_name to Unknown 

wine_data.taster_name.fillna('Unknown',inplace=True)



# Setting the NaN variety to Unknown 

wine_data.variety.fillna('Unknown',inplace=True)



# Setting the NaN winery to Unknown 

wine_data.winery.fillna('Unknown',inplace=True)

def proportionate(row, total):

    

    proportion = float("{0:.3f}".format((row.amount/(total))*100))

    return str(proportion) + '%'

#countries_proportion = pd.DataFrame({'Country': wine_data.country, 'Count': a})

countries_proportion = wine_data.groupby('country').description.agg([len]).sort_values(by='len', ascending=False)



countries_proportion = countries_proportion.rename_axis("", axis='rows').rename_axis("country", axis='columns').rename(columns={'len': 'amount'})

# 

total = countries_proportion.amount.sum()

countries_proportion['proportion']= countries_proportion.apply(lambda row: proportionate(row,total), axis = 1)



countries_proportion.head()



# Set the width and height of the figure

plt.figure(figsize=(14,8))



# Add title

plt.title("Número de Vinhos Produzidos por País")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=countries_proportion.index[0:15], y=countries_proportion['amount'][0:15], palette="pastel")

sns.set_style("whitegrid")

# Add label for vertical axis

plt.ylabel("Países")

plt.xlabel("Quantidade de vinhos produzidos")
taster_name_proportion = wine_data.groupby('taster_name').description.agg([len]).sort_values(by='len', ascending=False)



taster_name_proportion = taster_name_proportion.rename_axis("", axis='rows').rename_axis("country", axis='columns').rename(columns={'len': 'amount'})

# 

total = taster_name_proportion.amount.sum()

taster_name_proportion['proportion']= taster_name_proportion.apply(lambda row: proportionate(row,total), axis = 1)



taster_name_proportion.head()
# Set the width and height of the figure

plt.figure(figsize=(14,8))



# Add title

plt.title("Quantidade de Avaliações por Avaliador")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(y=taster_name_proportion.index, x=taster_name_proportion['amount'], palette="pastel")

sns.set_style("whitegrid")

# Add label for vertical axis

plt.ylabel("Avaliadores")

plt.xlabel("Quantidade de avaliações")
variety_proportion = wine_data.groupby('variety').description.agg([len]).sort_values(by='len', ascending=False)



variety_proportion = variety_proportion.rename_axis("", axis='rows').rename_axis("country", axis='columns').rename(columns={'len': 'amount'})

# 

total = variety_proportion.amount.sum()

variety_proportion['proportion']= variety_proportion.apply(lambda row: proportionate(row,total), axis = 1)



variety_proportion.head()
# Set the width and height of the figure

plt.figure(figsize=(14,8))



# Add title

plt.title("Quantidade de Tipos de Uva Usados na Fabricação")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(y=variety_proportion.index[0:30], x=variety_proportion['amount'][0:30], palette="pastel")

sns.set_style("whitegrid")

# Add label for vertical axis

plt.ylabel("Países")

plt.xlabel("Quantidade de vinhos produzidos")
price_score = wine_data.describe()

price_score = price_score.style.set_precision(10)

price_score
sns.distplot(a=wine_data['score'], kde=False)
sns.regplot(x=wine_data['price'], y=wine_data['score'],lowess = True)
def is_US(row):

    if row.country == 'US':

        return 'US'

    else:

        return 'not US'





wine_US = wine_data.copy()

wine_US['country'] = wine_US.apply(lambda row: is_US(row), axis = 1)



wine_US.head()
#sns.scatterplot(x=wine_US['price'], y=wine_US['score'], hue=wine_US['country'])

sns.lmplot(y="price", x="score", hue="country", data=wine_US)

def cost_benefit(row):

    return row.score/row.price



def classificate(row):

    if row.cost_benefit <= 2.12:

        return 'Low cost-benefit'

    elif row.cost_benefit <= 3.46:

        return 'Medium cost-benefit'

    elif row.cost_benefit <= 5.12:

        return 'High cost-benefit'

    else:

        return 'Very high cost-benefit'

wine_data['cost_benefit'] = wine_data.apply(lambda row: cost_benefit(row), axis = 1)

wine_data.cost_benefit.describe()



wine_data['CB classification'] = wine_data.apply(lambda row: classificate(row), axis = 1)



wine_data.head()
def classificate_cb(row):

    if row.cost_benefit_mean <= 2.12:

        return 'Low cost-benefit'

    elif row.cost_benefit_mean <= 3.46:

        return 'Medium cost-benefit'

    elif row.cost_benefit_mean <= 5.12:

        return 'High cost-benefit'

    else:

        return 'Very high cost-benefit'



group = wine_data.groupby(['country']).agg({'score': ['mean', 'max', 'min'], 'price': ['mean','min'], 'cost_benefit': 'mean'})

group.columns = ['_'.join(col).strip() for col in group.columns.values]

group = group.sort_values(by='score_mean', ascending = False)



group['CB classification'] = group.apply(lambda row: classificate_cb(row), axis = 1)



group.head()
normalized = group[['score_mean', 'price_mean','cost_benefit_mean']].transform(lambda x: x/x.max())

normalized = normalized.reset_index()

normalized.head()


df = pd.DataFrame(np.c_[normalized.price_mean[0:8],normalized.cost_benefit_mean[0:8],normalized.score_mean[0:8]], index=normalized.country[0:8])

df.plot.bar(figsize= (10,8), title = 'Classificação dos países de acordo com preço médio, custo benefício e pontuação')





plt.legend(['Preço Médio','Custo Benefício Médio','Pontuação Média'],loc='best')



plt.show()
best_wines = wine_data[(wine_data['score'] == 100)]

best_wines.sort_values(by = 'price', ascending = True).head()

best_wine = wine_data.iloc[best_wines.price.idxmin(), :]

best_wine