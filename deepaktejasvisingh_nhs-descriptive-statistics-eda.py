# Import libraries



import re

import seaborn as sns

import matplotlib.pyplot as plt

import plotly

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/expenditure-in-the-salisbury-nhs-v2/expenditure_v2.csv')

df.head()
# Check data types



df['Expenditure'].dtypes
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.isna().sum()
# Cast to string



df["Expenditure"] = df["Expenditure"].astype(str)
# Parse string and remove non-numeric characters 



def make_decimal(x):

    non_decimal = re.compile(r'[^\d.]+')

    non_decimal = non_decimal.sub('', x)

    

    return non_decimal
df["Expenditure"] = df["Expenditure"].apply(make_decimal)
df.head()
# Check al correctly converted



df.isna().sum()
df['Expenditure'] = pd.to_numeric(df['Expenditure'], errors='coerce')
# Check all correctly converted again



df.isna().sum()
# Describe the columns

    

for col in df.columns:

    print ("There are {} unique values in the {} column.\n".format(len(df[col].unique()), col))
# Remove unnecessary columns



df.drop(["Department Family", "Entity"], axis=1, inplace=True)
df.head()
len(df)
# See what NaN rows are like



df.loc[df["Expense Area"].isna() == True]
df.fillna("Unknown", inplace=True)
df.isna().sum()
df["Year"] = df['Date'].dt.year

df.head()
# Calculate by expenditure and by transaction number





# by Expense Type

df_x_type = df.groupby(['Expense Type']).agg({'Expenditure': 'sum'})

df_x_type["% of Total Expenditure"] = df_x_type.apply(lambda x: (100 * x) / float(x.sum()))

df_x_type["No. of Transactions"] = df['Expense Type'].value_counts()

df_x_type["% of Total Transactions"] = (df['Expense Type'].value_counts(normalize=True) * 100)

df_x_type.sort_values("Expenditure", ascending = False, inplace=True)



# By Expense Area

df_x_area = df.groupby(['Expense Area']).agg({'Expenditure': 'sum'})

df_x_area["% of Total Expenditure"] = df_x_area.apply(lambda x: (100 * x) / float(x.sum()))

df_x_area["No. of Transactions"] = df['Expense Area'].value_counts()

df_x_area["% of Total Transactions"] = (df['Expense Area'].value_counts(normalize=True) * 100)

df_x_area.sort_values("Expenditure", ascending = False, inplace=True)



# By supplier

df_supplier = df.groupby(['Supplier']).agg({'Expenditure': 'sum'})

df_supplier["% of Total Expenditure"] = df_supplier.apply(lambda x: (100 * x) / float(x.sum()))

df_supplier["No. of Transactions"] = df['Supplier'].value_counts()

df_supplier["% of Total Transactions"] = (df['Supplier'].value_counts(normalize=True) * 100)

df_supplier.sort_values("Expenditure", ascending = False, inplace=True)
df_x_type.head()
df_x_area.head()
df_supplier.head()
fig, ax = plt.subplots(3, 1, figsize=(7,15))



sns.scatterplot(x=df_x_type["% of Total Transactions"], y=df_x_type["% of Total Expenditure"], ax=ax[0])

sns.scatterplot(x=df_x_area["% of Total Transactions"], y=df_x_area["% of Total Expenditure"], ax=ax[1])

sns.scatterplot(x=df_supplier["% of Total Transactions"], y=df_supplier["% of Total Expenditure"], ax=ax[2])

plt.show()
fig, ax = plt.subplots(3, 1, figsize=(15,15))



sns.barplot(x=df_x_type.head().index, y="Expenditure", data=df_x_type.head(), ax=ax[0])

sns.barplot(x=df_x_area.head().index, y="Expenditure", data=df_x_area.head(), ax=ax[1])

sns.barplot(x=df_supplier.head().index, y="Expenditure", data=df_supplier.head(), ax=ax[2])
df.to_csv('all_data.csv', index=False)

df_x_type.to_csv('numbers_by_expense_type.csv', index=False)

df_x_area.to_csv('numbers_by_expense_area.csv', index=False)

df_supplier.to_csv('numbers_by_supplier.csv', index=False)