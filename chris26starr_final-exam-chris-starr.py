# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#machine learning imports
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import os
print(os.listdir("../input"))
# Read the dataset 
abalone = pd.read_csv('../input/abalone-dataset/abalone.csv')
abalone
abalone.columns
abalone.hist(figsize=(20,10),bins=9)
plt.figure(figsize=(15,9))
sns.countplot(x='Sex',data=abalone)
ulimit = np.percentile(abalone['Length'].values, 95)
llimit = np.percentile(abalone['Length'].values, 5)
filtered = abalone[(abalone['Length']<ulimit) & (abalone['Length']>llimit)]

filtered.hist(figsize=(20,10),bins=30)
categorical = ['Sex']
X = abalone.drop(categorical,axis=1)
scaler = StandardScaler()
lb = LabelBinarizer()
X = scaler.fit_transform(X)
X = np.c_[X,lb.fit_transform(abalone['Sex'])]
X_train,X_test,y_train,y_test = train_test_split(X,abalone['Rings'])
reg = DecisionTreeRegressor()
reg.fit(X_train,y_train)
reg.score(X_test,y_test)
orders = pd.read_csv("../input/brazilian-ecommerce/olist_orders_dataset.csv",parse_dates=[('order_purchase_timestamp'),('order_delivered_customer_date')])
orders
orders['purchase_month'] = orders['order_purchase_timestamp'].dt.month
orders
plt.figure(figsize=(15,9))
sns.countplot(x='purchase_month',data=orders)
reviews = pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")
reviews
order_reviews= orders.merge(reviews, on='order_id')
order_reviews
avg_score = order_reviews.groupby("purchase_month").mean()['review_score']
avg_score
plt.figure(figsize=(15,9))
sns.barplot(x=avg_score.index,y=avg_score)
review_dataset=order_reviews.groupby("order_status").size()
review_dataset['avg_score'] = order_reviews.groupby("order_status").mean()['review_score']

review_dataset
oproducts = pd.read_csv("../input/brazilian-ecommerce/olist_products_dataset.csv")
oproducts.head()
pcname = pd.read_csv("../input/brazilian-ecommerce/product_category_name_translation.csv")
pcname.head()
joined = oproducts.merge(pcname, on='product_category_name')
joined.head()
ooitems = pd.read_csv("../input/brazilian-ecommerce/olist_order_items_dataset.csv")
ooitems.head()
second_join = ooitems.merge(joined, on='product_id')
second_join.head()
final_join= order_reviews.merge(second_join, on='order_id')
final_join.head()
top_ten = final_join.groupby("product_category_name_english").mean()['review_score'].sort_values(ascending=False)[:10]
top_ten
