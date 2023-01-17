# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Machine Learning

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Activation, Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam



# Data Visualization

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS



import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv")







fig, ax = plt.subplots(figsize=(12, 5))



sns.barplot(x = 'keyword', y = 'count', data = df[:20], ax = ax)



ax.set(xlabel='Keyword', ylabel='Frequency')

plt.xticks(rotation=45, ha='right')



plt.show()
df = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")

df.dataframeName = "European Wish Data"
df.head(5)
df.shape
df.columns
# get the number of missing data points per column

df.isnull().sum()
df = df.dropna(axis=0, subset=['rating_five_count', 

                               'rating_four_count', 

                               'rating_three_count', 

                               'rating_two_count', 

                               'rating_one_count'])
print(df['tags'].head(5))
df['tags'] = df['tags'].apply(lambda tag: tuple(val.lower() for val in tag.split(',')))



print(df['tags'].head(5))
print("Unique values: ", df['product_color'].unique())

print("Value counts:")

print(df['product_color'].value_counts())
color_map = {'leopardprint' : 'pattern', 'navyblue' : 'blue', 'beige' : 'brown', 'lightblue' : 'blue',

             'armygreen' : 'green', np.nan : 'unknown', 'khaki' : 'brown', 'red&blue' : 'twocolor', 

             'blue&pink' : 'twocolor', 'white&green' : 'twocolor', 'winered' : 'red', 'black&green' : 'twocolor',

             'whitefloral' : 'pattern', 'floral' : 'pattern', 'fluorescentgreen' : 'green','orange&camouflage' : 'pattern',

             'lightyellow' : 'yellow', 'coolblack' : 'black', 'multicolor' : 'pattern', 'camouflage' : 'pattern', 

             'lightpink' : 'pink', 'pink&black' : 'twocolor', 'silver' : 'grey', 'lightgreen' : 'green', 

             'mintgreen' : 'green', 'pink&grey' : 'twocolor', 'gray' : 'grey', 'coffee' :'brown', 'rose' : 'red',

             'leopard' : 'pattern', 'black&white' : 'twocolor', 'orange-red' : 'orange', 'dustypink' : 'pink', 

             'star' : 'pattern', 'white&black' : 'twocolor', 'apricot' : 'orange', 'skyblue' : 'blue', 

             'burgundy' : 'red', 'claret' : 'red', 'pink&white' : 'twocolor', 'rosered' : 'red', 'lightred' : 'red', 

             'coralred' : 'red', 'lakeblue' : 'blue', 'darkblue' : 'blue', 'camel' : 'brown','pink&blue' : 'twocolor',

             'nude' : 'brown', 'lightpurple' : 'purple', 'army' : 'pattern', 'black&stripe' : 'twocolor',

             'greysnakeskinprint' : 'pattern', 'denimblue' :  'blue', 'applegreen' : 'green', 'offwhite' : 'white',

             'lightgray' : 'grey', 'navy' : 'blue', 'gray&white' : 'twocolor', 'brown&yellow' : 'twocolor',

             'winered&yellow' : 'twocolor', 'whitestripe' : 'white', 'rainbow' : 'pattern', 'lightgrey' : 'grey',

             'watermelonred' : 'red', 'prussianblue' : 'blue', 'navyblue&white' : 'twocolor', 'white&red' : 'twocolor',

             'wine' : 'red', 'ivory' : 'white', 'black&yellow' : 'twocolor', 'jasper' : 'green', 'lightkhaki' : 'brown',

             'offblack' : 'black', 'violet' : 'purple', 'black&blue' : 'twocolor', 'blackwhite' : 'twocolor', 

             'rosegold' : 'pink', 'gold' : 'yellow'}



df['product_color'] = df['product_color'].str.lower()

df['product_color'] = df['product_color'].str.replace(' ', '')



df['product_color'] = df['product_color'].replace(color_map)



print("Unique values: ", df['product_color'].unique())

print("Value counts:")

print(df['product_color'].value_counts())
print("Unique values: ", df['product_variation_size_id'].unique())

print("Value counts:")

print(df['product_variation_size_id'].value_counts())
size_map = {'Size-XS' : 'XS', 'M.' : 'M', np.nan : 'unknown', 'S.' : 'S', 's' : 'S', 'choose a size' : 'unknown',

            'XS.' : 'XS', '32/L' : 'L', 'Suit-S' : 'S', 'XXXXXL' : '5XL', 'EU 35' : 'unknown',  '4' : 'XS', 'Size S.' : 'S',

            '1m by 3m' : 'unknown', 'Size S' : 'S', 'Women Size 36' : 'unknown', 

            'US 6.5 (EU 37)' : 'unknown', 'XXXS' : '3XS', 'SIZE XS' : 'XS', '26(Waist 72cm 28inch)' : 'unknown',

            'Size XXS' : 'XXS', '29' : 'unknown', '1pc' : 'unit', '100 cm' : 'unknown', 'One Size' : 'unknown',

            'SIZE-4XL' : '4XL', '1' : 'unknown', 'S/M(child)' : 'unknown', '2pcs' : 'unit', 'XXXL' : '3XL', 

            'S..' : 'S', '30 cm' : 'unknown', '33' : 'unknown', 'Size M' : 'M', '100 x 100cm(39.3 x 39.3inch)' : 'unknown',

            '100pcs' : 'unit', '2XL' : 'XXL', 'SIZE XXS' : 'XXS', 'Base & Top & Matte Top Coat' : 'unknown', 

            'size S' : 'S', '35' : 'unknown', '34' : 'unknown', 'SIZE-XXS' : 'XXS', 'S(bust 88cm)' : 'S', 

            'S (waist58-62cm)' : 'S', 'S(Pink & Black)' : 'S', '20pcs' : 'unit', 'US-S' : 'S', 'Size -XXS' : 'XXS', 

            'X   L' : 'XL', 'XXXXL' : '4XL', '25' : 'unknown', 'SizeL' : 'L', 'Size-S' : 'S', 'Round' : 'unknown', 

            'Pack of 1' : 'unit', 'S Diameter 30cm' : 'unknown', 'AU plug Low quality' : 'unknown', '5PAIRS' : 'unit', 

            '25-S' : 'S', 'Size/S' : 'S', 'S Pink' : 'S', 'Size-5XL' : '5XL', 'daughter 24M' : 'M', '2' : 'unknown',

            'Baby Float Boat' : 'unknown', '10 ml' : 'unknown', '60' : 'unknown', 'Size-L' : 'L', 'US5.5-EU35' : 'unknown',

            '10pcs' : 'unit', '17' : 'unknown', 'Size-XXS' : 'XXS', 'Women Size 37' : 'unknown', 

            '3 layered anklet' : 'unknown', '4-5 Years' : 'unknown', 'Size4XL' : '4XL', 'first  generation' : 'unknown',

            '80 X 200 CM' : 'unknown', 'EU39(US8)' : 'unknown', 'L.' : 'L', 'Base Coat' : 'unknown', '36' : 'unknown',

            '04-3XL' : '3XL', 'pants-S' : 'S', 'Floating Chair for Kid' : 'unknown', '20PCS-10PAIRS' : 'unknown', 

            'B' : 'unknown', 'Size--S' : 'S', '5' : 'unknown', '1 PC - XL' : 'XL', 'H01' : 'unknown', '40 cm' : 'unknown',

            'SIZE S' : 'S'}



df['product_variation_size_id'] = df['product_variation_size_id'].replace(size_map)



print("Unique values: ", df['product_variation_size_id'].unique())

print("Value counts:")

print(df['product_variation_size_id'].value_counts())
print("Unique values: ", df['has_urgency_banner'].unique())

print("Value counts:")

print(df['has_urgency_banner'].value_counts())
df['has_urgency_banner'] = df['has_urgency_banner'].replace(np.nan, 0)



print("Unique values: ", df['has_urgency_banner'].unique())

print("Value counts:")

print(df['has_urgency_banner'].value_counts())
print("Unique values: ", df['urgency_text'].unique())

print("Value counts:")

print(df['urgency_text'].value_counts())
urgency_map = {"Quantité limitée !":"limitedquantity", "Réduction sur les achats en gros":"wholesalediscount",np.nan:"notext"}



df['urgency_text'] = df['urgency_text'].replace(urgency_map)



print("Unique values: ", df['urgency_text'].unique())

print("Value counts:")

print(df['urgency_text'].value_counts())
print("There were %d duplicate rows." % (df.duplicated().sum()))



df = df.drop_duplicates()
duplicate_features = ['title_orig', 'merchant_id', 'product_id']



print("There were %d rows with duplicate %s features." % (df.duplicated(subset=duplicate_features).sum(), tuple(duplicate_features)))

df[duplicate_features].where(df.duplicated(subset=duplicate_features) == True).dropna(axis=0)
df = df.drop_duplicates(subset=duplicate_features)
df.info()
df.shipping_option_name.value_counts(normalize=True)
df = df.drop(columns=['shipping_option_name'])
df.inventory_total.value_counts(normalize=True)
df = df.drop(columns=['inventory_total'])
df.origin_country.value_counts(normalize=True)
df = df.drop(columns=['origin_country'])
df.badges_count.value_counts(normalize=True)
df = df.drop(columns=['badges_count'])
df.badge_local_product.value_counts(normalize=True)
df = df.drop(columns=['badge_local_product'])
df.badge_fast_shipping.value_counts(normalize=True)
df = df.drop(columns=['badge_fast_shipping'])
df.shipping_is_express.value_counts(normalize=True)
df = df.drop(columns=['shipping_is_express'])
df = df.drop(columns = ['title', 'currency_buyer', 'merchant_title', 'merchant_name', 

                        'merchant_info_subtitle', 'merchant_has_profile_picture', 

                        'merchant_profile_picture', 'product_url', 'theme', 

                        'crawl_month'])
df.info()
df['has_urgency_banner'] = df['has_urgency_banner'].astype(int)
numerical_features = ['price', 'retail_price', 'rating', 'rating_count', 'rating_five_count', 'rating_four_count', 

                      'rating_three_count', 'rating_two_count', 'rating_one_count', 'product_variation_inventory', 

                      'countries_shipped_to', 'merchant_rating_count', 'merchant_rating']



categorical_features = ['units_sold', 'uses_ad_boosts', 'badge_product_quality', 'product_color', 

                        'product_variation_size_id', 'shipping_option_price','has_urgency_banner',

                        'urgency_text']



other_features = ['title_orig', 'tags', 'merchant_id', 'product_picture', 'product_id']
hist = df[numerical_features].hist(figsize=(12, 12))



plt.tight_layout()

plt.show()
df.describe().transpose()
fig, ax = plt.subplots(figsize=(12, 5))



hist = sns.countplot(x='units_sold', data=df, order=sorted(df['units_sold'].unique()), ax=ax)



plt.xticks(rotation=45, ha='right')

plt.show()
fig = plt.figure(figsize=(12, 3))



boolean_features = ['uses_ad_boosts', 'badge_product_quality', 'has_urgency_banner']

for i in range(3):

    true_percentage = (df[boolean_features[i]].value_counts()[1] / len(df[boolean_features[i]])) * 100

    print("Percent that the '%s' flag is True: %f" % (boolean_features[i], true_percentage))

    

    fig.add_subplot(1, 3, i + 1)

    sns.countplot(df[boolean_features[i]])

    

plt.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(12, 5))



sns.countplot(x='shipping_option_price', data=df, order=sorted(df['shipping_option_price'].unique()), ax=ax)



ax.set(xlabel='shipping_option_price', ylabel='count')

plt.xticks(rotation=45, ha='right')



plt.show()
fig, ax = plt.subplots(figsize=(12, 5))



sns.countplot(x='product_color', data=df, order=df['product_color'].value_counts().index, ax=ax)



ax.set(xlabel='product_color', ylabel='count')

plt.xticks(rotation=45, ha='right')



plt.show()
fig, ax = plt.subplots(figsize=(12, 5))



size_order = ['unknown', 'unit', '3XS', 'XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL', '4XL', '5XL', '6XL']



sns.countplot(x='product_variation_size_id', data=df, order=size_order, ax=ax)



plt.xticks(rotation=45, ha='right')

plt.show()
fig, ax = plt.subplots(figsize=(12, 3))



print(df['urgency_text'].value_counts())



sns.countplot(x='urgency_text', data=df, order=sorted(df['urgency_text'].unique()), ax=ax)



plt.xticks(rotation=45, ha='right')

plt.show()
# Visualizing product prices (prediction target).



fig = plt.figure(figsize=(12, 4))



fig.add_subplot(2,1,1)

sns.distplot(df['price'])



fig.add_subplot(2,1,2)

sns.boxplot(df['price'])





plt.tight_layout()
# Visualizing product prices (prediction target).



fig = plt.figure(figsize=(12, 5))



fig.add_subplot(2,1,1)

ax1 = sns.distplot(df['price'])

ax1.set_xlim(-10, 260)



fig.add_subplot(2,1,2)

ax2 = sns.distplot(df['retail_price'])

ax2.set_xlim(-10, 260)



plt.tight_layout()
(df['retail_price'] - df['price']).value_counts()
def is_successful(units_sold):

    return 1 if units_sold >= 10000 else 0
df['is_successful'] = df['units_sold'].apply(is_successful)



print('Percent of successful products: %f' % ((df['is_successful'].value_counts()[1] / len(df['is_successful'])) * 100))



sns.countplot(df['is_successful'])



plt.show()
df_successful = df.where(df['is_successful'] == True).drop('is_successful', axis=1).dropna(axis=0)



df_successful[['title_orig', 'price', 'retail_price', 'units_sold', 'rating', 'merchant_rating']].head(5)
df_successful.describe().transpose()
fig = plt.figure(figsize=(12, 4))



fig.add_subplot(2,1,1)

sns.distplot(df_successful['price'])



fig.add_subplot(2,1,2)

sns.boxplot(df_successful['price'])



plt.tight_layout()
fig = plt.figure(figsize=(12, 5))



fig.add_subplot(2,1,1)

ax1 = sns.distplot(df_successful['price'])

ax1.set_xlim(-10, 260)



fig.add_subplot(2,1,2)

ax2 = sns.distplot(df_successful['retail_price'])

ax2.set_xlim(-10, 260)



plt.tight_layout()
fig = plt.figure(figsize=(12, 9))



boolean_features = ['uses_ad_boosts', 'badge_product_quality', 'has_urgency_banner']

for i in range(3):

    fig.add_subplot(3, 1, i + 1)

    

    sns.distplot(df_successful['price'].where(df_successful[boolean_features[i]] == 1), bins=10, label='%s = True' % boolean_features[i])

    sns.distplot(df_successful['price'].where(df_successful[boolean_features[i]] == 0), bins=10, label='%s = False' % boolean_features[i])

    

    plt.legend()

    

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(12, 12))



sns.heatmap(df_successful.corr(), annot=True, fmt='.2f', annot_kws={'size': 7.5}, square=True)



plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(12, 12))



k = 10

cols = df_successful.corr().nlargest(k, 'price')['price'].index

cm = np.corrcoef(df_successful[cols].values.T)



sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 7.5}, square=True, yticklabels=cols.values, xticklabels=cols.values)



plt.tight_layout()

plt.show()
features = ['price', 'shipping_option_price', 'retail_price', 'product_variation_inventory', 'merchant_rating']
fig = plt.figure()



sns.pairplot(df_successful[features])



plt.tight_layout()

plt.show()