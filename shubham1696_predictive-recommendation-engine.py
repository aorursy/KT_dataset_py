# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
customersDF = pd.read_csv("../input/brazilian-ecommerce/olist_customers_dataset.csv")

geolocationDF = pd.read_csv("../input/brazilian-ecommerce/olist_geolocation_dataset.csv")

order_itemsDF = pd.read_csv("../input/brazilian-ecommerce/olist_order_items_dataset.csv")

order_paymentsDF = pd.read_csv("../input/brazilian-ecommerce/olist_order_payments_dataset.csv")

order_reviewsDF = pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")

ordersDF = pd.read_csv("../input/brazilian-ecommerce/olist_orders_dataset.csv")

productsDF = pd.read_csv("../input/brazilian-ecommerce/olist_products_dataset.csv")

sellersDF = pd.read_csv("../input/brazilian-ecommerce/olist_sellers_dataset.csv")

product_category_name_translation = pd.read_csv("../input/brazilian-ecommerce/product_category_name_translation.csv")
customersDF.info()

geolocationDF.info()

order_itemsDF.info()

order_paymentsDF.info()

sellersDF.info()

product_category_name_translation.info()
order_reviewsDF.info()

order_reviewsDF.drop(columns = ['review_comment_title','review_comment_message','review_creation_date','review_answer_timestamp'],inplace=True)

order_reviewsDF.head()
order_reviewsDF.info()

order_reviewsDF.describe()
ordersDF.info()

ordersDF.drop(columns = ['order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date'],inplace=True)

ordersDF.head()
ordersDF.info()

ordersDF.describe()
productsDF.info()

print("\n")

print(productsDF.shape[0])

indexOfNullProductCategory = productsDF[productsDF['product_category_name'].isnull()].index

productsDF.drop(inplace=True, index=indexOfNullProductCategory)
productsDF.info()

print("\n")

print(productsDF.shape[0])

indexOfNullAdditional = productsDF[productsDF['product_weight_g'].isnull()].index

productsDF.drop(inplace=True, index=indexOfNullAdditional)

print("\n")

print(productsDF.shape[0])

print("\n")

productsDF.info()
productsDF.describe()
masterDF = ordersDF.copy()

masterDF = masterDF.merge(customersDF,on='customer_id',indicator = True)

masterDF = masterDF.merge(order_reviewsDF,on='order_id')

masterDF = masterDF.merge(order_paymentsDF,on='order_id')

masterDF = masterDF.merge(order_itemsDF,on='order_id')

masterDF = masterDF.merge(productsDF,on='product_id')

masterDF = masterDF.merge(sellersDF,on='seller_id')

masterDF.head()
masterDF.shape
masterDF.isnull().sum()
masterDF.info()
popular_products = pd.DataFrame(masterDF.groupby('product_id')['review_score'].count())

most_sold = popular_products.sort_values('review_score', ascending=False)

most_sold.head(30).plot(kind = "bar")
highestRated = pd.DataFrame(masterDF.groupby('product_id').agg(

    review_score_Avg = ('review_score', 'mean'),

    review_score_Count = ('review_score', 'count')

    ))



highestRated.sort_values(['review_score_Avg','review_score_Count'],ascending=False,inplace=True)           

highestRated.head(30)
import matplotlib.pyplot as plt



# %matplotlib inline

plt.style.use("ggplot")





import sklearn

from sklearn.decomposition import TruncatedSVD