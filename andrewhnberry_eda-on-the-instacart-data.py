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
#Importing the holy trinity of data science packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#Other Visualization Packages

import seaborn as sns
#Loading in our data 

df_orders = pd.read_csv("/kaggle/input/instacart-market-basket-analysis/orders.csv")

df_products = pd.read_csv("/kaggle/input/instacart-market-basket-analysis/products.csv")

df_aisles = pd.read_csv("/kaggle/input/instacart-market-basket-analysis/aisles.csv")

df_dept = pd.read_csv("/kaggle/input/instacart-market-basket-analysis/departments.csv")



df_ord_pro_train = pd.read_csv("/kaggle/input/instacart-market-basket-analysis/departments.csv")

df_ord_pro_prior = pd.read_csv("/kaggle/input/instacart-market-basket-analysis/order_products__prior.csv")
df_orders.head()
df_products.head()
df_aisles.head()
df_dept.head()
df_ord_pro_train.head()
df_ord_pro_prior.head()
plt.figure(figsize = (14,7))

sns.countplot(x='order_hour_of_day', data= df_orders)

plt.title('Number of Orders Taken by Hour of the Day.')

plt.ylabel('Count')

plt.xlabel('Hour')

plt.show()
plt.figure(figsize = (14,7))

sns.countplot(x='order_dow', data= df_orders)

plt.title('Number of Orders Taken by Day of the Week.')

plt.ylabel('Count')

plt.xlabel('Day')

plt.show()
agg_dow_hour = df_orders.groupby(['order_hour_of_day', 'order_dow'])['order_number'].aggregate('count').reset_index()

agg_dow_hour = agg_dow_hour.pivot('order_hour_of_day','order_dow','order_number')



plt.figure(figsize =(14,7))

sns.heatmap(agg_dow_hour)

plt.title('Heatmap of orders for Hour of the day Vs. Day of the Week')

plt.show()
plt.figure(figsize = (14,7))

sns.countplot(x='days_since_prior_order', data= df_orders)

plt.title('Days Since Prior Instacart Order')

plt.ylabel('Count')

plt.xlabel('Days')

plt.show()
df_ord_pro_prior = pd.merge(df_ord_pro_prior, df_products, on = 'product_id', how = 'left')

df_ord_pro_prior = pd.merge(df_ord_pro_prior, df_aisles, on = 'aisle_id', how = 'left')

df_ord_pro_prior = pd.merge(df_ord_pro_prior, df_dept, on = 'department_id', how='left')



df_ord_pro_prior = df_ord_pro_prior.drop(['product_id', 'aisle_id', 'department_id'], axis = 1)



df_2 = df_ord_pro_prior.copy()



df_2.head(10)
print(f'Thee are {df_2.product_name.nunique()} unique products sold on Instacart! Wow!')
top15_products = df_2.product_name.value_counts()[:15]
plt.figure(figsize = (14,7))

top15_products.plot(kind = 'bar', color = 'limegreen')

plt.title('Top 15 Products sold on Instacart', fontsize = 20)

plt.ylabel('Count')

plt.xlabel('Product Name')

plt.xticks(rotation = 30)

plt.show()
top15_aisles = df_2.aisle.value_counts()[:15]
plt.figure(figsize = (14,6))

sns.barplot(top15_aisles.index, top15_aisles.values, color = 'deepskyblue')

#Turns of scientic Notation in plot

plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)

plt.title('Top 15 Aisles Shopped on Instacart', fontsize = 20)

plt.ylabel('Count')

plt.xlabel('Aisle Name')

plt.xticks(rotation = 45)

plt.show()
dept_fre_count = df_2.department.value_counts()

dept_percentage = np.array(dept_fre_count/ dept_fre_count.sum())*100

dept_name = np.array(dept_fre_count.index)

plt.figure(figsize = (12,12))

plt.pie(dept_percentage, labels = dept_name, autopct = '%1.1f%%')

plt.title('Pie Chart for the vairous departments')

plt.show()