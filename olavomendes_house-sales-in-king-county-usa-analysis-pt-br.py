import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import HeatMap



plt.style.use("seaborn-muted")

sns.set_style('darkgrid')

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 200)

pd.set_option('display.float_format', lambda x: '%.5f' % x)



%matplotlib inline
raw_data = pd.read_csv('../input/housesalesprediction/kc_house_data.csv', parse_dates=['date', 'yr_built'])
raw_data.head(10)
print('Dimensão dos dados')

print('Linhas:',raw_data.shape[0])

print('Colunas:',raw_data.shape[1])
raw_data.info()
raw_data = raw_data.assign(year_built=raw_data.yr_built.dt.year)

raw_data.drop('yr_built', 1, inplace=True)
raw_data.head()
pd.DataFrame(raw_data['price'].describe())
med_price = raw_data['price'].median()



plt.figure(figsize=[20, 8])

sns.distplot(raw_data['price'], color = 'r', label = 'Distribuição')

plt.axvline(med_price, color='b', linestyle='dashed', label='Mediana')

plt.ticklabel_format(style='plain', axis='x')

plt.title('Distribuição do preço de vendas das casas')

plt.xlabel('Preço (em dólar)')

plt.ylabel('Densidade')

plt.xticks(np.arange(raw_data['price'].min(), raw_data['price'].max(), step=500000))



plt.legend()

plt.show()
raw_data.isnull().sum().sort_values(ascending=False)
plt.figure(figsize=[12, 6])



plt.subplot(121)

sns.boxplot(x='condition', y='price', data=raw_data);



plt.subplot(122)

sns.boxplot(x='grade', y='price', data=raw_data)



plt.show()
pd.DataFrame(raw_data['condition'].value_counts().sort_index())
pd.DataFrame(raw_data['grade'].value_counts().sort_index())
var_num = raw_data._get_numeric_data()

var_num.drop('id', 1, inplace=True)

var_num.head()
var_num.corr()
var_num_corr = var_num.corr()



plt.figure(figsize = [12, 8])

sns.heatmap(var_num_corr, vmin=-1, vmax=1, linewidth=0.01, linecolor='black', cmap='RdBu_r')

plt.show()
var_num_corr['price'].sort_values(ascending=False).round(3)
cols = raw_data[['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'sqft_above',

                         'sqft_living15', 'price']]

most_corr_var = cols.corr()



plt.figure(figsize=[10, 6])

sns.heatmap(data=most_corr_var, vmin=-1, vmax=1, linewidth=0.01, linecolor='black', cmap='RdBu_r', annot=True)



plt.show()
plt.figure(figsize=[15, 15])



i = 1



for col in cols:

    if col == 'price':

        continue

    plt.subplot(4, 2, i)

    sns.regplot(raw_data[col], raw_data['price'], line_kws={'color': 'r'})

    plt.xlabel(col)

    plt.ylabel('Preço de venda ($)')

    i+=1

    



plt.tight_layout()

plt.show()
raw_data['view'].value_counts() / len(raw_data)
plt.figure(figsize=(8, 4))

sns.set_style("darkgrid")



sns.regplot(raw_data['view'], raw_data['price'], line_kws={'color': 'r'})



plt.show()
plt.figure(figsize=(12, 6))



sns.scatterplot(raw_data['waterfront'], raw_data['price'], hue=raw_data['waterfront'])



plt.show()
plt.figure(figsize=(12, 6))



cmap = sns.cubehelix_palette(dark=.8, light=.3, as_cmap=True)

sns.scatterplot(raw_data['floors'], raw_data['price'], hue=raw_data['floors'], palette=cmap)



plt.show()
dates = pd.DataFrame(raw_data['year_built'], columns=['year_built'])

bins = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2015]

labels = ['1900 - 1910', '1911 - 1920', '1921 - 1930', '1931 - 1940', '1941 - 1950', '1951 - 1960', '1961 - 1970', '1971 - 1980',

         '1981 - 1990', '1991 - 2000', '2001 - 2010', '2011 - 2015']

raw_data['decade_built'] = pd.cut(dates['year_built'], bins, labels = labels, include_lowest = True)

raw_data.sample(5)
plt.figure(figsize=[12, 8])



sns.barplot(x=raw_data['price'], y=raw_data['decade_built'], palette="cubehelix_d")



plt.show()
prices = pd.DataFrame(raw_data['price'], columns=['price'])

bins = [0, 250000, 500000, 1000000, 8000000]

labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

raw_data['price group'] = pd.cut(prices['price'], bins, labels = labels, include_lowest = True)

raw_data.sample(5)
raw_data.groupby('price group')['price group'].count()
raw_data = raw_data.assign(renovated=(raw_data['yr_renovated'] > 0).astype(int))
raw_data.head(10)
renovated = raw_data.groupby('renovated')['price'].count()

renovated
renovated_median = raw_data.groupby('renovated')['price'].median()

renovated_median
plt.figure(figsize=(15, 5))

        

plt.subplot(1, 2, 1)

plt.pie(renovated, explode = (0, 0.1), colors=['r', 'lightblue'], labels= ['Não', 'Sim'], autopct='%1.1f%%')

plt.title('A casa foi renovada ou não?')



plt.subplot(1, 2, 2)

sns.barplot(x=renovated.index, y = renovated_median, palette=['r', 'lightblue'])

plt.title('Mediana dos preços das casas renovadas ou não')

plt.ylabel('Mediana de preço')



plt.tight_layout()

plt.show()