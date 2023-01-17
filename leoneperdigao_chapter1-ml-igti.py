import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # improved UI for charts

import matplotlib.pyplot as plt

import google

# read dataframe

customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
# print dataframe sample

customers.head()
# print dataframe metadata

customers.info()
# sum all null values

customers.isnull().sum()
# create null values randomly

customers_null=customers

for col in customers_null.columns:

    customers_null.loc[customers_null.sample(frac=0.1).index, col] = np.nan
# print customers_null metadata

customers_null.info()
# print customers_null sample

customers_null.head()
# sum null values

customers_null.isnull().sum()
# dropping rows with null values

customers_null.dropna()
# Fill null values with 0

customers_null.fillna(0)
# Print dataframe stats

customers_null.describe()
# Fill null values with column average value

customers_null.fillna(customers_null.mean())
customers.describe()
boxplot = customers.boxplot(column=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
# checking for anomalies

from scipy import stats

z = np.abs(stats.zscore(customers['Annual Income (k$)'].values))

threshold = 2

result = np.where(z > threshold)

df_incoming_outlier = customers.iloc[result[0]]
df_incoming_outlier
# analyzing the distribution of customers by gender

sns.countplot(x='Gender', data=customers)

plt.title('Customers distribution by gender')
customers.hist('Age', bins=35)

plt.title('Customers distribution by age')

plt.xlabel('Age')
cat_df_customers = customers.select_dtypes(include=['object'])
cat_df_customers.head()