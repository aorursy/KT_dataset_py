#Importing required libraries

import numpy as np

import pandas as pd

from pandas import DataFrame,Series



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
#Filtering warnings out

import warnings

warnings.filterwarnings("ignore")
#Setting values for plots

#plt.rcParams['figure.figsize'] = (20,10)

plt.style.use('ggplot')
#import datasets

data = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')

cat = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.csv')

cat_count = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv')
data.info()
sns.kdeplot(data['price'],shade=True)

sns.kdeplot(data['retail_price'],shade=True)

plt.title('Price vs Retail Price')
data['currency_buyer'].unique()
sns.countplot('uses_ad_boosts',data=data)

plt.title('Ad-boosts')
corr = data['uses_ad_boosts'].corr(data['units_sold'])

print(f'The Correlation coeff between Ad-Boost and Units-Sold is {np.round(corr,4)}')
fig = sns.FacetGrid(data,hue='uses_ad_boosts',aspect=4)



fig.map(sns.kdeplot,'units_sold',shade=True)

max_units = data['units_sold'].max()

fig.set(xlim=(0,max_units))

fig.add_legend()

plt.title('Ad-boosts vs Sale')
rating_df = DataFrame(data[['rating', 'rating_count',

       'rating_five_count', 'rating_four_count', 'rating_three_count',

       'rating_two_count', 'rating_one_count','units_sold']])

sns.heatmap(rating_df.corr(),cmap='CMRmap_r',annot=True)

plt.title('Rating vs Sale')
badges_df = DataFrame(data[['badges_count',

       'badge_local_product', 'badge_product_quality', 'badge_fast_shipping','units_sold']])

sns.heatmap(badges_df.corr(),cmap='CMRmap_r',annot=True)

plt.title('BAdges vs Sale')
"""Creating a DF of Color vs Total-Units Sold"""

color = data[['product_color','units_sold']]

color_df = DataFrame(color.groupby(['product_color']).sum().sort_values('units_sold',ascending=False))

color_df.plot(kind='bar',figsize=(20,10))

plt.title('Color vs Sale')
"""Creating a DF of Size vs Total-Units Sold"""

size = data[['product_variation_size_id','units_sold']]

size_df = DataFrame(size.groupby(['product_variation_size_id']).sum().sort_values('units_sold',ascending=False))

size_df.plot(kind='bar',cmap='plasma',figsize=(20,10))

plt.title('Size vs Sale')
"""Label-Encoding the color and size"""

from sklearn.preprocessing import LabelEncoder
def encoder(value):

    encode = LabelEncoder().fit(value)

    return (encode.transform(value))



color_df = color_df.reset_index()

size_df = size_df.reset_index()



color_df['product_color'] = encoder(color_df['product_color'])

size_df['product_variation_size_id'] = encoder(size_df['product_variation_size_id'])
print('The relation btw color and units sold:')

print(color_df.corr(),end='\n\n\n')

print('The relation btw size and units sold:')

print(size_df.corr())
sns.lmplot('units_sold','product_variation_inventory',data=data)

plt.title('Variation Inventory vs Sale')
sns.lmplot('units_sold','inventory_total',data=data)

plt.title('Total inventory vs Sale')
ship = data.groupby('shipping_option_name')['shipping_option_name'].count()

plt.pie(ship,radius=2)

plt.legend(ship.index,loc=(-0.9,0.3))

plt.title('Shipping Name???')
ship = data.groupby('shipping_is_express')['shipping_is_express'].count()

lables = ship.index

plt.pie(ship,labels=lables,colors=['brown','pink'])

plt.legend(['0:No','1:Yes'])

plt.title('Express Shipping???')
sns.violinplot('shipping_option_price',data=data)

plt.title('Shipping Price')
ship_price = data[['shipping_option_price','units_sold']]

sns.heatmap(ship_price.corr(),annot=True)

plt.title('Shipping price vs Sale')
plt.rcParams['figure.figsize'] = (20,10)

data['countries_shipped_to'].plot(kind='hist',color='purple')

plt.title(' # of Destination countries')
sns.scatterplot('countries_shipped_to','units_sold',data=data)

plt.title('Destination Countries vs Sale')
print('A negative correlation exists between units sold and countries shipped with a value')

print(data['countries_shipped_to'].corr(data['units_sold']))
urgency = data[['has_urgency_banner', 'urgency_text','units_sold']]

urgency = urgency.replace(np.nan,0)
urgency['has_urgency_banner'].count() == urgency['urgency_text'].count()
fig = sns.FacetGrid(urgency,hue='has_urgency_banner',aspect=4)



fig.map(sns.kdeplot,'units_sold')

x_max = urgency['units_sold'].max()

fig.set( xlim = (0,x_max))

fig.add_legend()

plt.title('Urgency banner vs Sale')
c = urgency['has_urgency_banner'].corr(urgency['units_sold'])

print(f'The correlation between the two is {c}')
merchant = data[['origin_country', 'merchant_title', 'merchant_name',

       'merchant_info_subtitle', 'merchant_rating_count', 'merchant_rating',

       'merchant_id', 'merchant_has_profile_picture',

       'merchant_profile_picture','units_sold']]
merchant.info()
sns.countplot('origin_country',data=merchant)

plt.title('Origin Country')
for i in ['merchant_name','merchant_info_subtitle']:

    y = merchant[i].isna()

    merchant[i] = y.apply(lambda x : 0 if x else 1)
print('Correlation btw "Merchant name" and "Units Sold:"')

print(merchant['merchant_name'].corr(merchant['units_sold']),end='\n\n')

print('Correlation btw "merchant_info_subtitle" and "Units Sold":')

print(merchant['merchant_info_subtitle'].corr(merchant['units_sold']))
sns.boxplot(merchant['merchant_rating_count'],color='yellow',showmeans=True)

plt.title('# of Merchant ratings')
merchant['merchant_rating_count'].describe().astype(int)
mean = merchant['merchant_rating_count'].mean()

merchant['merchant_rating_count'] = merchant['merchant_rating_count'].apply(lambda x: mean if x>24564 else x)

sns.boxplot(merchant['merchant_rating_count'],color='yellow',showmeans=True)

fig = plt.gcf()

fig.set_size_inches(10,5)

plt.title('Rating Count')
sns.boxplot(merchant['merchant_rating'],color='green',showmeans=True)

fig = plt.gcf()

fig.set_size_inches(10,5)

plt.title('Rating')
df1 = merchant[['merchant_rating_count','merchant_rating','units_sold']]

sns.heatmap(df1.corr(),annot=True)

fig = plt.gcf()

fig.set_size_inches(10,5)

plt.title('Rating vs Sale')
merchant['merchant_has_profile_picture'] = merchant['merchant_has_profile_picture'].apply(lambda x : 'yes' if x==1 else 'no')

fig = sns.FacetGrid(merchant,hue='merchant_has_profile_picture',aspect=4)



fig.map(sns.kdeplot,'units_sold')

x_max = urgency['units_sold'].mean() # Considering the Mean sales level

fig.set( xlim = (0,x_max))

fig.add_legend()

plt.title('Merchant Profile pic vs Sale')