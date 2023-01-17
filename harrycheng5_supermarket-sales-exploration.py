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
df.info()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df['Date'] = pd.to_datetime(df['Date'])

total_sales = df.groupby(['Branch', 'City']).agg({'Total': 'sum'})
total_sales = total_sales.reset_index()
print(total_sales)

plt.figure(figsize=(10,6))
ax = sns.barplot(x='Branch', y='Total', data=total_sales)
ax.set(xlabel='', ylabel='');
ax.set_title('Total Revenue by Branches', fontsize=20);
sns.set(font_scale = 3)

sales_byproduct = df.groupby(['Product line']).agg({'Total': 'sum', 'Quantity': 'count'})
print(sales_byproduct)

plt.figure(figsize=(28,8))
plt.subplot(1, 2, 1)
ax1 = sns.barplot(x='Total', y=sales_byproduct.index, data=sales_byproduct);
ax1.set_title('Sales by Product Lines', fontsize=30);
ax1.set(xlabel='', ylabel='');
for p in ax1.patches:
    width = p.get_width()
    ax1.text(width -1.5  ,
            p.get_y()+p.get_height()/2. + 0.2,
            '{:1.2f}'.format(width),
            ha="center")

plt.subplot(1, 2, 2)
ax2 = sns.barplot(x='Quantity', y=sales_byproduct.index, data=sales_byproduct);
ax2.set_title('Quantity by Product Line', fontsize=30);
ax2.set(xlabel='', ylabel='');
ax2.set_yticks([]);
for p in ax2.patches:
    width = p.get_width()
    ax2.text(width -1.5  ,
            p.get_y()+p.get_height()/2. + 0.2,
            '{:1.2f}'.format(width),
            ha="center")


n_gender = df['Gender'].value_counts()
n_type = df['Customer type'].value_counts()
gender_type = df.groupby(['Gender', 'Customer type']).agg({'Customer type': 'count'})
gender_type = gender_type.rename(columns={'Customer type': 'Quantity'})
n_gender
n_type
gender_type
payment = df['Payment'].value_counts()
payment
gender_payment = df.groupby(['Gender', 'Payment']).agg({'Quantity':'count'})
gender_payment['%'] = gender_payment.groupby(level=0).transform(lambda x: (x / x.sum()).round(2))
gender_payment
sales_bymonth = df.groupby(df['Date'].dt.month)['Total'].sum()
sales_bymonth
branch_sales = df.groupby(['Branch', df['Date'].dt.month])['Total'].sum()
print(branch_sales)

branch_sales = branch_sales.reset_index()
ax = sns.catplot(x='Date', y='Total', data=branch_sales, col='Branch', kind='bar');
ax.set(xlabel='Month', ylabel='Sales');
rating_bybranch = df.groupby('Branch')['Rating'].mean()
rating_bybranch
