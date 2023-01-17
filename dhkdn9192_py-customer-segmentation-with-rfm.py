import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

%matplotlib inline
data = pd.read_excel('Online_Retail.xlsx')

data.shape
data.head()
data.tail()
data.info()
# filtering null CustomerID

data = data[~data['CustomerID'].isnull()]

data.shape
filtered_data = data[['Country', 'CustomerID']].drop_duplicates()

filtered_data.shape
filtered_data.Country.value_counts()[:10].plot(kind='bar')
uk_data = data[data.Country=='United Kingdom']

uk_data.info()
uk_data.describe()