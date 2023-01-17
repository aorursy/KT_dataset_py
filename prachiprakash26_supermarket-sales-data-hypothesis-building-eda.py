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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Reading the csv data
supermarket_sales = pd.read_csv('/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv')
# Checking the columns and type
supermarket_sales.info()
# Checking for number of rows and columns
supermarket_sales.shape
# Checking for top 5 records
supermarket_sales.head()
# Statistical information of numerical columns
supermarket_sales.describe()
# Check for missing values
supermarket_sales.isnull().sum()
# Check for duplicate values
supermarket_sales.duplicated().sum()
# Converting Date object to datetime
supermarket_sales['Date'] = pd.to_datetime(supermarket_sales['Date'], format = '%m/%d/%Y')
supermarket_sales.head()
# Checking the unique City, Customer Type, Gender, Product Line, Payments
print('City: ', supermarket_sales['City'].unique())
print('Branch: ',supermarket_sales['Branch'].unique())
print('Customer Type: ',supermarket_sales['Customer type'].unique())
print('Gender: ',supermarket_sales['Gender'].unique())
print('Product Line: ',supermarket_sales['Product line'].unique())
print('Payment: ',supermarket_sales['Payment'].unique())
# Deriving Month and Year from Date 
supermarket_sales['Month'] = supermarket_sales['Date'].dt.month
supermarket_sales['Year'] = supermarket_sales['Date'].dt.year
print('Year: ',supermarket_sales['Year'].unique())
print('Month: ',supermarket_sales['Month'].unique())
# a) Food and beverages are the most sold product line
sns.countplot(supermarket_sales['Product line'],
              order = supermarket_sales['Product line'].value_counts().index)
plt.xticks(rotation=45)
# Conclusion: Hypothesis is false as Fashion accessories are most sold product line
# b) Female buyers are greater than male buyers
sns.countplot(supermarket_sales['Gender'],
              order = supermarket_sales['Gender'].value_counts().index)
plt.xticks(rotation=45)
# Conclusion: Hypothesis is true as Female buyers are more
# c) Cash is the most convinient mode of payment
sns.countplot(supermarket_sales['Payment'],
              order = supermarket_sales['Payment'].value_counts().index)
plt.xticks(rotation=45)
# Conclusion: Hypothesis is false as Ewallet is preferred more than cash or credit card
# d) January month has the most purchase in 2019
sns.countplot(supermarket_sales['Month'],
              order = supermarket_sales['Month'].value_counts().index)
plt.xticks(rotation=45)
# Conclusion: Hypothesis is true followed by March and February
# e) There is a linear relationship between Total and Price
sns.scatterplot(x=supermarket_sales['Unit price'], y=supermarket_sales['Total'])
# Conclusion: The hypothesis is true as the relationship seen is linear
# Plotting numerical value vs date
supermarket_sales[['Date','Total']].set_index('Date').plot()
plt.xlabel('Date')
plt.ylabel('Total')
plt.title('Plot of Total Vs Date')
# Date Wise Analysis
supermarket_sales[supermarket_sales['Month'] == 1][['Date','Total']].set_index('Date').plot()
plt.xlabel('Date')
plt.ylabel('Total')
plt.title('January Month Total Distribution')
supermarket_sales[supermarket_sales['Month'] == 2][['Date','Total']].set_index('Date').plot()
plt.xlabel('Date')
plt.ylabel('Total')
plt.title('February Month Total Distribution')
supermarket_sales[supermarket_sales['Month'] == 3][['Date','Total']].set_index('Date').plot()
plt.xlabel('Date')
plt.ylabel('Total')
plt.title('March Month Total Distribution')
# Histogram for numerical data
numerical_cols = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross margin percentage', 'gross income', 'Rating']
supermarket_sales[numerical_cols].hist(bins=15, figsize=(25, 10), layout=(3, 3));
# Boxplot for numerical data
supermarket_sales[numerical_cols].boxplot()
plt.xticks(rotation=45)
# Barplot for categorical data
sns.countplot(supermarket_sales['Customer type'],
              order = supermarket_sales['Customer type'].value_counts().index)
plt.xticks(rotation=45)
# Product Line Vs Total boxplot
sorted_salesdata = supermarket_sales.groupby(['Product line'])['Total'].median().sort_values()
sns.boxplot(x='Product line', y='Total', data=supermarket_sales, order=list(sorted_salesdata.index))
plt.xticks(rotation=90)
# City Vs Total boxplot
sorted_salesdata = supermarket_sales.groupby(['City'])['Total'].median().sort_values()
sns.boxplot(x='City', y='Total', data=supermarket_sales, order=list(sorted_salesdata.index))
plt.xticks(rotation=90)
# Crating scatterplot for categorical data for Gender
sns.catplot(x="Gender", y="Total", data=supermarket_sales)
# Distribution of Product Line for Gender
sns.catplot(x="Product line", y="Total", data=supermarket_sales, hue='Gender')
plt.xticks(rotation=90)
# Distribution of Product Line for Payment
sns.catplot(x="Product line", y="Total", data=supermarket_sales, hue='Payment')
plt.xticks(rotation=90)
# Plot showing the combination of scatter plot along with histogram
sns.jointplot(x=supermarket_sales['Unit price'], y=supermarket_sales['gross income'])
# Creating a correlation heatmaps
plt.figure(figsize=(8, 8))
sns.heatmap(supermarket_sales[numerical_cols].corr(), annot=True)
# Bonus : What is the average rating of the supermarket
supermarket_sales['Rating'].mean()
# Bonus : What is the average rating City wise
supermarket_sales.groupby(['City'])['Rating'].mean()