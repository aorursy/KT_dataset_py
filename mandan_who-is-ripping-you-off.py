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
import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")
amazon = pd.read_csv('../input/amazon.csv')

amazon.head()
flipkart = pd.read_csv('../input/flipkart.csv')

flipkart.head()
# dropping the outliers.



wild_card = flipkart[flipkart.flipkart_price > 1000].index



flipkart = flipkart.drop(wild_card)

amazon = amazon.drop(wild_card)
plt.figure(figsize=(10, 6))

sns.distplot(flipkart.flipkart_price, color='red', label="Flipkart", kde=False, bins=30)

sns.distplot(amazon.amazon_price, color='green', label="Amazon", kde=False, bins=30)

plt.legend()

plt.xlabel('Price')

plt.ylabel('Number of Books')
flipkart.columns = ['flipkart_author', 'isbn', 'flipkart_title',

       'flipkart_ratings count', 'flipkart_price', 'flipkart_stars']



amazon.columns = ['amazon_title', 'amazon_author', 'amazon_rating',

       'amazon_reviews count', 'isbn', 'amazon_price']



df = pd.merge(flipkart, amazon, how='outer')

df = df.drop_duplicates()
df.head()
cols_to_drop = ['flipkart_ratings count','flipkart_stars', 'amazon_title', 'amazon_author',

       'amazon_rating', 'amazon_reviews count']

df.drop(cols_to_drop, axis=1, inplace=True)

df.columns = ['author', 'isbn', 'title', 'flipkart_price', 'amazon_price']
df.tail()
f_minus_a = df['flipkart_price'] - df['amazon_price']

a_minus_f = df['amazon_price'] - df['flipkart_price']
df['price_diff'] = df['flipkart_price'] - df['amazon_price']
df.price_diff.describe()
df['price_diff_perc'] = df['price_diff']/df['flipkart_price']*100

df.tail()
df.price_diff_perc.describe()
max_price = []

min_price = []

l = list()

for i, row in df.iterrows():

    if row['flipkart_price'] < row['amazon_price']:

        max_price.append(row['amazon_price'])

        min_price.append(row['flipkart_price'])

    else:

        max_price.append(row['flipkart_price'])

        min_price.append(row['amazon_price'])


print()

print("If you only bought books from Flipkart: {}".format(df['flipkart_price'].sum()))



print()

print("If you only bought books from Amazon: {}".format(df['amazon_price'].sum()))
plt.figure(figsize=(10, 6))

sns.kdeplot(max_price, color='red')

sns.kdeplot(min_price, color='green')
print("Price Worst Case Scenario: {}".format(sum(max_price)))

print("Price Best Case Scenario: {}".format(sum(min_price)))



print()

diff = sum(max_price) - sum(min_price)

print("Difference: {}".format(diff))

print("Percentage difference: {0:.2f}%".format(diff/sum(max_price)*100))
plt.figure(figsize=(10, 6))

sns.distplot(abs(df['price_diff_perc']), kde=False, color='green', bins=100)

plt.xlabel('Price Difference (Percentage)')
no_diff = df[df['price_diff'] == 0].count()[0]/df.shape[0]

print('Percentage of books which are priced the same: {0:.2f}%'.format(no_diff*100))



print()

no_diff = df[(abs(df['price_diff_perc']) > 0) & (abs(df['price_diff_perc']) < 100)].count()[0]/df.shape[0]

print('Percentage of books with price difference: {0:.2f}%'.format(no_diff*100))



print()

no_diff = df[df['price_diff_perc'] < 0].count()[0]/df.shape[0]

print('Better pricing on Flipkart: {0:.2f}%'.format(no_diff*100))



print()

no_diff = df[df['price_diff_perc'] > 0].count()[0]/df.shape[0]

print('Better pricing on Amazon: {0:.2f}%'.format(no_diff*100))
# How much would you overpay if you choose any one of the site, compared to our best price?



print("Only Flipkart: {0:.2f}%".format((100*(df.flipkart_price.sum() - sum(min_price))/sum(min_price))))

print()

print("Only Amazon: {0:.2f}%".format((100*(df.amazon_price.sum() - sum(min_price))/sum(min_price))))

plt.figure(figsize=(10, 6))

sns.distplot(df['price_diff_perc'], kde=False, color='green')

plt.xlabel('Price Difference (Percentage)')

plt.ylabel('Number of books')
# Books less than 500

plt.figure(figsize=(10, 6))

sns.distplot(df[df['flipkart_price'] <500 ]['price_diff_perc'], kde=False, color='red')

plt.xlabel('Price Difference (Percentage)')

plt.ylabel('Number of books')
# Books greater than 500

plt.figure(figsize=(10, 6))

sns.distplot(df[df['flipkart_price'] > 500]['price_diff_perc'], kde=False, color='blue')

plt.xlabel('Price Difference (Percentage)')

plt.ylabel('Number of books')