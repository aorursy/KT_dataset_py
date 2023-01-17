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
df = pd.read_csv('/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv')
df.head()
df.dtypes
df.info()
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Time'] = pd.to_datetime(df['Time'])
df['Hour'] = df['Time'].dt.hour
import seaborn as sns
sns.set_style("darkgrid")
gendercount = sns.countplot(x = 'Gender', data = df).set_title('Gender Count')
sales_branch = sns.countplot(x = 'Branch', data = df).set_title('Sales count by Branch')
rating_branch = sns.boxplot(x = 'Branch', y = 'Rating', data = df).set_title('Ratings by Branch')
sns.boxplot(x = 'Customer type', y = 'Rating', data = df).set_title('Ratings by Members or Non-members')
df1 = df[['Branch', 'Product line', 'Quantity']]
df2 = df1.groupby(['Branch', 'Product line']).sum().reset_index()

df2_plot = sns.catplot(x = 'Product line', y = 'Quantity', hue = 'Branch', data = df2, kind = 'bar', height = 8, aspect = 2)
df2_plot.set_xticklabels(rotation = 90)
cus_type = sns.countplot(x = 'Customer type', data = df).set_title('Customer type')
mt = df[['Customer type', 'Total']]
mt = mt.groupby(['Customer type']).mean().reset_index()
mt.columns = ['Customer type', 'Average spending']
mt
sns.barplot(x = 'Customer type', y = 'Average spending', data = mt)
payment = sns.countplot(x = 'Payment', data = df).set_title('Payment method')
quan_sales = sns.lineplot(x = df['Hour'], y = 'Quantity', data = df)
hspm_branch = sns.relplot(x = 'Hour', y = 'Quantity', row = 'Month', col = 'Branch',hue = 'Gender', data = df, kind = 'line', col_order = ('A', 'B', 'C'), height = 3)
MvP = sns.countplot(x = 'Payment', hue = 'Customer type', data =df)
sns.countplot(x = 'Hour', hue = 'Customer type', data = df)
sns.countplot(x = 'Gender', hue = 'Customer type', data = df)
import matplotlib.ticker as ticker
sales_plot = sns.lineplot(x = 'Date', y = 'gross income', data = df)
