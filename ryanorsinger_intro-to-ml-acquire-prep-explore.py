import numpy as np # linear algebra library

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Import our visualization libraries

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Import scipy's statistics ttest for independence

from scipy.stats import ttest_ind

import scipy.stats as stats



# Import our machine learning models from sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import pearsonr



# ignore warning

import warnings

warnings.filterwarnings("ignore")
# Acquire Step (Our data sources may be in multiple places, in multiple databases, etc...)

# Our data here is straightforward

df = pd.read_csv("raw_lemonade_data.csv") # df is short for dataframe and dataframes are how we get all the data into one variable.

print(df.head())

print(df.tail())
# Ensure that the Date column is a proper date data-type instead of a string or a number

df["Date"] = pd.to_datetime(df["Date"])
# Let's calculate revenue as price times sales

df["Revenue"] = df["Price"] * df["Sales"]

df.head()
# Looks like we need to clean the data for the price column.. Real data can get pretty messy. This is only a preview!

df["Price"] = df.Price.str.replace("$", "").replace(" ", "") # Remove all of the dollar signs and a hidden extra space in the price column

df.Price = df.Price.astype(np.float64)



df.Revenue = df.Price * df.Sales 
df = df.set_index(df['Date']) 

df = df.drop("Date", 1) # drop the old Date column

df.head()
# Let's start exploring by visualizing the data!

sales_over_time = df.Revenue.plot()

sales_over_time.set(xlabel='Day', ylabel='Dollars in Revenue', title='Revenue over a year') 
sns.relplot(x='Rainfall', y='Revenue', col='Price', data=df) # There's not a linear relationship, but the curve is clear. More rain and less revenue tend to move together (correlation not causation)
# Explore how Temperature relates to Revenue

sns.relplot(x='Temperature', y='Revenue', data=df) 
sns.relplot(x='Flyers', y='Revenue', data=df)
with sns.axes_style('white'):

    j = sns.jointplot("Temperature", "Revenue", data=df, kind='reg', height=5);

    j.annotate(stats.pearsonr)

plt.show()
with sns.axes_style('white'):

    j = sns.jointplot("Flyers", "Revenue", data=df, kind='reg', height=5);

    j.annotate(stats.pearsonr)

plt.show()
with sns.axes_style('white'):

    j = sns.jointplot("Rainfall", "Revenue", data=df, kind='reg', height=5);

    j.annotate(stats.pearsonr)

plt.show()
sns.heatmap(df.corr(), cmap='Blues', annot=True)
df = df.drop(columns=["Price", "Sales"])
sns.pairplot(df)
sns.heatmap(df.corr(), cmap='Blues', annot=True)