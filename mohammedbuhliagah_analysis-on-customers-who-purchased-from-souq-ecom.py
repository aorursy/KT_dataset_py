import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#reading the datasets
Athletic_Shoes = pd.read_csv('/kaggle/input/ecommerce-analysis/Athletic_Shoes.csv')

Casual_Shoes = pd.read_csv('/kaggle/input/ecommerce-analysis/Casual_Shoes.csv')

eyewear = pd.read_csv('/kaggle/input/ecommerce-analysis/eyewear.csv')

Perfumes_and_Fragrances = pd.read_csv('/kaggle/input/ecommerce-analysis/Perfumes_and_Fragrances.csv')

Sportswear = pd.read_csv('/kaggle/input/ecommerce-analysis/Sportswear.csv')

Wallets = pd.read_csv('/kaggle/input/ecommerce-analysis/Wallets.csv')

watches = pd.read_csv('/kaggle/input/ecommerce-analysis/watches.csv')
#shape of all data sets
(Athletic_Shoes.shape, Casual_Shoes.shape,  eyewear.shape, Perfumes_and_Fragrances.shape, Sportswear.shape,

    Wallets.shape, watches.shape)
Athletic_Shoes.head(2)
Casual_Shoes.head(2)
eyewear.head(2)
Perfumes_and_Fragrances.head(2)
Sportswear.head(2)
Wallets.head(2)
watches.head(2)
pdList = [Athletic_Shoes, Casual_Shoes,  eyewear, Perfumes_and_Fragrances, Sportswear,

    Wallets, watches]  # List of dataframes

df = pd.concat(pdList)
df.isnull().sum()
df.isnull().sum().sum()
df.dtypes
df.head(2)
#remove the , and SAR and change the type to float
df['item_price'] = df['item_price'].str.replace('[,SAR]', '').astype(float)
df['item_after_discount'] = df['item_after_discount'].str.replace('[,SAR]', '').astype(float)
df.head(2)
df.dtypes
df.columns = ['category', 'product', 'Original_price', 'price', 'perc_of_unit_sold', 'rating', 'shpping_rate', 'gender']
df.head(2)
df.to_csv('combined_datasets.csv')
df.describe()
# exploaring the higher spender from both gender
spending  = df.groupby('gender').price.sum()
spending
spending.plot(kind='bar', figsize=(18,7));

plt.title('Amount spending per gender', fontsize=18);
#above average purchase per category
above_average_sale = df.groupby('category').perc_of_unit_sold.mean()
above_average_sale
above_average_sale.plot(kind='bar', figsize=(18,7));

plt.title('Average spending on Purchased per Category ', fontsize=18);
X  = df.groupby(['shpping_rate', 'rating']).perc_of_unit_sold.mean()
X
X.plot(kind='bar', figsize=(18,7), color='blue');

plt.title('Average of purchases upon free shiping and customer review', fontsize=18);