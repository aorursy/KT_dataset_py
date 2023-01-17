import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
order_products_train_df = pd.read_csv("../input/order_products__train.csv")

order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")

orders_df = pd.read_csv("../input/orders.csv")

products_df = pd.read_csv("../input/products.csv")

aisles_df = pd.read_csv("../input/aisles.csv")

departments_df = pd.read_csv("../input/departments.csv")
plt.figure(figsize=(6,6))

dframe =pd.value_counts(orders_df['eval_set'].values, sort=False)

dframe.plot(kind='bar',legend=None,title="Total counts in prior, test and train dataset")
dframe =pd.value_counts(orders_df['order_hour_of_day'].values, sort=False)

dframe.plot(kind='bar')
dframe =pd.value_counts(orders_df['order_dow'].values, sort=False)

dframe.plot(kind='bar')
grouped_df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_number"].count().reset_index()

grouped_df.head()

grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')

grouped_df.head()
plt.figure(figsize=(20,12))

sns.heatmap(grouped_df, linewidths=.5, cmap='cool') #cbar = False

aisles_df.head()
df2 =pd.merge(order_products_prior_df, products_df, on='product_id', how='inner')

df3 = pd.merge(df2, aisles_df, on='aisle_id', how='inner')

df3.head()
product_id_count =pd.crosstab(index=df2["product_name"],  # Make a crosstab

                      columns="count") 

product_id_count.columns = ['frequency']

df =product_id_count.sort_values('frequency', ascending=False)

df_new= df.head(20)

df_new.plot(kind='bar')

product_id_count2 =pd.crosstab(index=df3["aisle"],  # Make a crosstab

                      columns="count") 

product_id_count2.columns = ['frequency']

df2_new =product_id_count2.sort_values('frequency', ascending=False)

df2_new2= df2_new.head(20)

df2_new2.plot(kind='bar')
from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot #some plotly libraries



print(__version__) # requires version >= 1.9.0