import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text

import graphviz





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
geo_data = pd.read_csv("../input/geolocation_olist_public_dataset.csv")

public_data = pd.read_csv("../input/olist_public_dataset_v2.csv")

customers_data = pd.read_csv("../input/olist_public_dataset_v2_customers.csv")

payments_data = pd.read_csv("../input/olist_public_dataset_v2_payments.csv")

product_data = pd.read_csv("../input/product_category_name_translation.csv")
geo_data
geo_data.info()
geo_data["city"].value_counts()
geo_data["state"].value_counts()
public_data
public_data.info()

public_data["order_id"].value_counts()
public_data["order_status"].value_counts()
public_data["order_items_qty"].value_counts()
public_data["order_sellers_qty"].value_counts()
public_data["order_purchase_timestamp"].value_counts()
public_data["order_delivered_customer_date"].value_counts()
customers_data
payments_data
product_data
left_public = public_data.set_index(['product_category_name'])

right_product = product_data.set_index('product_category_name')

combined_data = left_public.join(right_product, lsuffix='_PUBLIC', rsuffix='_PRODUCT')



left_combined_data = combined_data.set_index(['order_id'])

right_payments = payments_data.set_index('order_id')

combined_data = left_combined_data.join(right_payments, lsuffix='_COM', rsuffix='_PAYMENTS')



left_combined_data = combined_data.set_index(['customer_id'])

right_customers = customers_data.set_index('customer_id')

combined_data = left_combined_data.join(right_customers, lsuffix='_COM', rsuffix='_CUST')

combined_data = combined_data.reset_index()
combined_data.groupby('customer_unique_id').order_items_qty.agg([sum, 'count']).sort_values(by='sum', ascending=False)
combined_data.groupby(['customer_state', 'customer_city']).order_items_qty.agg([sum, 'count']).sort_values(by=['sum', 'customer_state'], ascending=False)
combined_data.groupby('product_category_name_english').order_items_qty.agg([sum, 'count', max]).sort_values(by=['sum'], ascending=False)
combined_data.groupby('product_category_name_english').order_items_qty.agg([max]).sort_values(by=['max'], ascending=False)
canceled_data = combined_data.loc[combined_data.order_status == 'canceled']

canceled_data.groupby('product_category_name_english').order_items_qty.agg([sum, 'count']).sort_values(by=['sum'], ascending=False)
combined_data.groupby('product_category_name_english').review_score.agg(['mean']).sort_values(by=['mean'], ascending=False)
combined_data.groupby('payment_type').order_items_qty.agg(['count']).sort_values(by=['count'], ascending=False)
combined_data.groupby('customer_unique_id').order_items_qty.agg(['count', sum]).sort_values(by=['sum'], ascending=False)
combined_data.groupby('customer_unique_id').order_products_value.agg([max]).sort_values(by='max', ascending=False)
combined_data.groupby(['customer_state', 'customer_city']).order_freight_value.agg([max]).sort_values(by=['max'], ascending=False)
combined_data.groupby(['customer_unique_id']).review_score.agg(['count']).sort_values(by=['count'], ascending=False)
combined_data.hist(edgecolor="black", linewidth=1.2, figsize=(30, 30));
combined_data = combined_data.iloc[:1000,:]

y = combined_data.product_category_name_english

combined_data_features = ['review_score', 'order_items_qty', 'order_products_value']

x = combined_data[combined_data_features]



missing_val_count_by_column = (x.isnull().sum())

missing_val_count_by_column
product_tree = DecisionTreeClassifier().fit(x.to_numpy(), y.to_numpy())

product_graph = export_graphviz(product_tree,

                               out_file=None,

                               feature_names = combined_data_features,

                               class_names = y,

                               special_characters=True,

                               rounded=True,

                               filled=True)

# View Graph

graphviz.Source(product_graph)
print(export_text(product_tree, feature_names=combined_data_features))