# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline

from plotnine import *



#import plotly.plotly as plt

#import cufflinks as cf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
my_data2=pd.read_excel("../input/sales_data.xlsx")

my_data2.head()
my_data1=pd.read_excel("../input/date.xlsx") 

my_data1.head()
my_data = pd.concat([my_data2, my_data1], axis=1, sort=False)

my_data.head()
# Check for missing values

my_data.isnull().sum()
# See rows with missing values

my_data[my_data.isnull().any(axis=1)]
# Plot histogram using seaborn

#plt.figure(figsize=(15,8))

#sns.distplot(my_data2['transaction timestamp'], bins =30)

#sns.distplot(my_data['transaction timestamp'],kde = False)

(my_data['transaction timestamp']).hist(bins =70, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)

# Add title and axis names

plt.title('Visualisation')

plt.xlabel('TRANSACTION TIMESTAMP')

plt.ylabel('TOTAL COUNT')
total = my_data.isnull().sum().sort_values(ascending=False)

percent = (my_data.isnull().sum()/my_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()
my_data.dropna(subset=['product description'], how='all', inplace = True)

my_data.dropna(subset=['customer id'], how='all', inplace = True)

my_data.isnull().sum()
my_data.head()
prod_sales_amt=(my_data['quantity sold']).mul(my_data['unit price'])

prod_sale_amt=prod_sales_amt.T 

prod_sale_amt = prod_sale_amt.to_frame()

prod_sale_amt.columns = ['prd sale amt']

prod_sale_amt.head()
my_data1 = pd.concat([my_data, prod_sale_amt], axis=1, join_axes=[my_data.index])

my_data1.head()
qty_sold_data = my_data1.nlargest(50, "quantity sold") 

unit_price_data = my_data1.nlargest(50, "unit price") 

prd_sale_amt= my_data1.nlargest(50,'prd sale amt')
plt.figure(figsize=(20,10))

my_data['day_name'].value_counts()[:10].plot.pie(autopct='%1.1f%%',

        shadow=True, startangle=90, cmap='tab20')

plt.title("MOST PERCENTAGE OF SALES IN A WEEK",size=14, weight='bold')
# Plot the total quantity sold cournty wise

sns.set_color_codes("pastel")

plt.figure(figsize=(15,10))

sns.barplot(x="quantity sold", y="transaction country", data=qty_sold_data,

            label="Country Wise Sales", color="b")
plt.figure(figsize=(15,10))

sns.set_color_codes("muted")

sns.barplot(x="quantity sold", y="product description", data=qty_sold_data,

            label="Product Wise Sales", color="g")
# Plot the total quantity sold cournty wise

sns.set_color_codes("pastel")

plt.figure(figsize=(15,10))

sns.barplot(x="month_of_year", y="quantity sold", data=qty_sold_data,

            label="Country Wise Sales", color="y")
my_data1.describe()
new_my_data= my_data1.filter(['quantity sold','unit price','customer id','day_of_month','prd sale amt'], axis=1)

new_my_data.head()
sns.pairplot(my_data1)
# RuntimeWarning: invalid value encountered in divide <to avoid this I am using np.seterr>

np.seterr(divide='ignore', invalid='ignore')

sns.pairplot(qty_sold_data, hue="quantity sold", palette="husl")
sns.pairplot(unit_price_data, hue="unit price", palette="husl")
sns.pairplot(prd_sale_amt, hue="prd sale amt", palette="husl")
sns.catplot(x="day_of_month", y="prd sale amt", hue="customer id", kind="swarm", data=qty_sold_data,height=10);
sns.catplot(x="day_of_month", y="prd sale amt", hue="customer id", kind="swarm", data=prd_sale_amt,height=10);
sns.catplot(x="month_of_year",y="prd sale amt",kind='box',data=qty_sold_data, size =10)
sns.catplot(x="month_of_year",y="prd sale amt",kind='box',data=prd_sale_amt, size =10)
col_names = ['day_of_month','customer id']



fig, ax = plt.subplots(len(col_names), figsize=(16,12))



for i, col_val in enumerate(col_names):



    sns.distplot(my_data[col_val], hist=True, ax=ax[i])

    ax[i].set_title('Freq dist '+col_val, fontsize=10)

    ax[i].set_xlabel(col_val, fontsize=8)

    ax[i].set_ylabel('Count', fontsize=8)



plt.show()
# Show the results of a linear regression within each dataset

sns.lmplot(x="day_of_month", y="quantity sold", col="unit price", hue="unit price", data=qty_sold_data,

           col_wrap=2, ci=None, palette="muted", height=4,

           scatter_kws={"s": 50, "alpha": 1})
# Show the results of a linear regression within each dataset

sns.lmplot(x="day_of_month", y="quantity sold", col="unit price", hue="unit price", data=unit_price_data,

           col_wrap=2, ci=None, palette="muted", height=4,

           scatter_kws={"s": 50, "alpha": 1})