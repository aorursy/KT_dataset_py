#data manipulation

import numpy as np

import pandas as pd

import re

from unicodedata import normalize



#data visualization

import matplotlib.pyplot as plt

import seaborn as sns



#silencing warnings

import warnings

warnings.filterwarnings("ignore")
shop_a  = pd.read_csv('../input/tshirt-prices-in-brazils-largest-retail-stores/shop_a')

shop_b  = pd.read_csv('../input/tshirt-prices-in-brazils-largest-retail-stores/shop_b')

shop_c  = pd.read_csv('../input/tshirt-prices-in-brazils-largest-retail-stores/shop_c')

shop_d  = pd.read_csv('../input/tshirt-prices-in-brazils-largest-retail-stores/shop_d')
sns.set_style('darkgrid')



my_palette = ['#8338ec',

              '#ff006e',

              '#02c39a',

              '#ffe74c',

              '#3a86ff']



color = '#8338ec'



sns.set_palette(my_palette, 5)
shop_a
shop_a_brands = pd.read_csv('../input/tshirt-prices-in-brazils-largest-retail-stores/shop_a_brands.csv',

                            header= None)

shop_a_brands = shop_a_brands[0].tolist()

shop_a_brands = map(str.lower, shop_a_brands)

shop_a_brands = ' | '.join(shop_a_brands)

shop_a_brands = str('( ' + shop_a_brands + ' )')



shop_a['brand_name'] = shop_a['brand_name'].apply(lambda x: x.lower())

brand = shop_a['brand_name'].str.extract(pat = str(shop_a_brands))

shop_a['brand_name'] = brand[0]



shop_a = shop_a[~shop_a['brand_name'].isna()]
shop_a.loc[(shop_a['final_price'].isna()), 'final_price'] = shop_a.loc[(shop_a['final_price'].isna()), 'original_price']
shop_a
shop_b
shop_b['brand_name'] = shop_b['brand_name'].apply(lambda x: x.lower())
shop_b.loc[(shop_b['final_price'].isna()), 'final_price'] = shop_b.loc[(shop_b['final_price'].isna()), 'original_price']
shop_b
shop_c
shop_c_brands = pd.read_csv('../input/tshirt-prices-in-brazils-largest-retail-stores/shop_c_brands.csv',

                            header= None)



#eliminate (xxx)

regex = re.compile('\(.*\)')

shop_c_brands[0] = shop_c_brands[0].apply(lambda x: re.sub(regex, '', x))

shop_c_brands = shop_c_brands[0].tolist()

shop_c_brands = map(str.lower, shop_c_brands)

shop_c_brands = ' | '.join(shop_c_brands)

shop_c_brands = str('( ' + shop_c_brands + ' )')



shop_c['brand_name'] = shop_c['brand_name'].apply(lambda x: x.lower())

brand = shop_c['brand_name'].str.extract(pat = str(shop_c_brands))

shop_c['brand_name'] = brand[0]



shop_c = shop_c[~shop_c['brand_name'].isna()]
shop_c.loc[(shop_c['original_price'].isna()), 'original_price'] = shop_c.loc[(shop_c['original_price'].isna()), 'final_price']
shop_c
shop_d
shop_d['brand_name'] = shop_d['brand_name'].apply(lambda x: x.lower())
shop_d.loc[(shop_d['final_price'].isna()), 'final_price'] = shop_d.loc[(shop_d['final_price'].isna()), 'original_price']
shop_d
ds = [shop_a, shop_b,

      shop_c, shop_d]



data = pd.concat(ds, ignore_index= True)
data
#eliminar os parenteses

regex = re.compile('[R$]|[ ]|\.')

data['original_price'] = data['original_price'].apply(lambda x: re.sub(regex, '', x))

data['final_price'] = data['final_price'].apply(lambda x: re.sub(regex, '', x))



data['original_price'] = data['original_price'].apply(lambda x: re.sub(',', '.', x))

data['final_price'] = data['final_price'].apply(lambda x: re.sub(',', '.', x))



data['original_price'] = data['original_price'].astype('float')

data['final_price'] = data['final_price'].astype('float')



data['original_price'] = data['original_price'].apply(lambda x: round(x, 2))

data['final_price'] = data['final_price'].apply(lambda x: round(x, 2))
def out_Ac(txt, codif='utf-8'):

    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')



data['brand_name'] = data['brand_name'].apply(lambda x: out_Ac(x))



data['brand_name'] = data['brand_name'].apply(lambda x: re.sub(' ', '', x))
#can i use a thesaurus

elimine = ['kit', 'regata', 'pacote', 'fardo',

           'pack', 'machao', 'machão', 'conjunto',

           'pçs', ]



for i in elimine:

    data.drop(data[data['brand_name'].str.contains(str(i))].index,

            inplace= True)



data.reset_index(inplace= True, drop= True)
value_counts = data['brand_name'].value_counts()



data = data.set_index('brand_name').join(value_counts)



threshold = 50



data = data[data['brand_name'] > threshold]



data['brand_name'] = data.index



data.reset_index(drop= True, inplace= True)
data.describe()
def elim_Outlier(data, col, q1, q3):

    iqr = (q3 - q1)

    out_min = q1 - (1.5 * iqr)

    out_max = q3 + (1.5 * iqr)



    return data[(data[col] >= out_min) &

                (data[col] <= out_max)]



data = elim_Outlier(data, 'original_price', 79, 129)

data = elim_Outlier(data, 'final_price', 54.9, 99)
nan_values = pd.DataFrame(data.isna().sum()).T

nan_values
len(data[data['original_price'] < data['final_price']])
len(data['brand_name'].unique())
def plotDistOthers(dataframe, column):

    sns.kdeplot(dataframe[column],

                shade= True)

    

    steps = int(dataframe[column].max() / 25)

    plt.xticks(range(int(dataframe[column].min()),

                     int(dataframe[column].max()) + steps,

                     steps),

               rotation= 90)



fig, ax = plt.subplots(figsize = (10, 12))



plt.subplot(2,1,1)

plotDistOthers(data, 'original_price')



plt.subplot(2,1,2)

plotDistOthers(data, 'final_price')



plt.subplots_adjust(hspace= 0.3)
rank_brand = data.groupby('brand_name').mean()



rank_brand_original = rank_brand.sort_values('original_price')

rank_brand_original = pd.concat([rank_brand_original.head(10),

                                 rank_brand_original.tail(10)])

rank_brand_original.reset_index(inplace= True)



rank_brand_final = rank_brand.sort_values('final_price')

rank_brand_final = pd.concat([rank_brand_final.head(10),

                              rank_brand_final.tail(10)])

rank_brand_final.reset_index(inplace= True)



def plotBarLocType(dataframe, col_x, col_y):

    sns.barplot(dataframe[col_y],

                dataframe[col_x],

                color= color)

    

    plt.xticks(range(0, 200, 10))

    

fig, ax = plt.subplots(figsize = (10, 12))



plt.subplot(2,1,1)

plotBarLocType(rank_brand_original, 'brand_name', 'original_price')



plt.subplot(2,1,2)

plotBarLocType(rank_brand_final, 'brand_name', 'final_price')



plt.subplots_adjust(hspace= 0.3)
fig, ax = plt.subplots(figsize = (10, 6))



sns.kdeplot(data['final_price'], data['original_price'], shade= True)



plt.subplots_adjust(hspace= 0.3)
data['discount_price_%'] = (1 - (data['final_price'] / data['original_price'])) * 100

data['discount_price_%'] = data['discount_price_%'].apply(lambda x: round(x, 2))



data['grade'] = 0



data.loc[(data['final_price'] >= 105), 'grade'] = '\$\$\$\$'

data.loc[(data['final_price'] < 105) & (data['final_price'] >= 85), 'grade'] = '\$\$\$'

data.loc[(data['final_price'] < 85) & (data['final_price'] >= 50), 'grade'] = '\$\$'

data.loc[(data['final_price'] < 50), 'grade'] = '\$'



data.sort_values('grade', inplace= True)



fig, ax = plt.subplots(figsize = (10, 6))



plt.subplot(1,1,1)

sns.scatterplot(data= data[data['discount_price_%'] > 0], x= 'original_price', y= 'discount_price_%', hue= 'grade')



plt.subplots_adjust(hspace= 0.3)