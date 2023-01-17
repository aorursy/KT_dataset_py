import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
original_data = pd.read_csv('/content/StockX-Data-Contest-2019-3.csv', header = 2)

df = original_data.copy()

df.head()
# Change 'order date' dtype

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')

df.head()
# Change 'release date' dtype

df['Release Date'] = pd.to_datetime(df['Release Date'], format='%m/%d/%Y')

df.head()
# Remove - from sneaker name

df['Sneaker Name'] = df['Sneaker Name'].apply(lambda x: x.replace('-', ' '))

df.head()
# Remove $ and comma from sale price

df['Sale Price'] = df['Sale Price'].apply(lambda x: x.replace('$', ''))

df['Sale Price'] = df['Sale Price'].apply(lambda x: x.replace(',', ''))

df.head()
# Remove $ from retail price

df['Retail Price'] = df['Retail Price'].apply(lambda x: x.replace('$', ''))

df.head()
df.to_csv('Clean_Shoe_Data.csv', index = False)