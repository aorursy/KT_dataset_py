# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
order_products_train_df = pd.read_csv("../input/order_products__train.csv")

order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")

orders_df = pd.read_csv("../input/orders.csv")

products_df = pd.read_csv("../input/products.csv")

aisles_df = pd.read_csv("../input/aisles.csv")

departments_df = pd.read_csv("../input/departments.csv")
products_df.head()
aisles_df.head()
departments_df.head()
products_details = pd.merge(left=products_df,right=departments_df,how="left")

products_details = pd.merge(left=products_details,right=aisles_df,how="left")

products_details.head()
plt.figure(figsize=(8,4))

g=sns.countplot(x="department",data=products_details)

g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")

plt.show()
plt.figure(figsize=(8,4))

top20_aisle=products_details["aisle"].value_counts()[:20].plot(kind="bar",title='Aisles')



fig,ax = plt.subplots(7,3, figsize=(20,45), gridspec_kw =  dict(hspace=1.4))

for i, j in enumerate(ax.flatten()):

    if(i < len(departments_df["department"])):

        data=products_details[products_details.department == departments_df.loc[i,"department"]].groupby(['aisle']).count()['product_id'].to_frame().reset_index()

        g = sns.barplot(data.aisle, data.product_id ,palette="Blues_d", ax=j)

        j.set_title('Dep: {}'.format(departments_df.loc[i,"department"]))

        j.set(xlabel = "Aisles", ylabel=" Number of products")

        g.set_xticklabels(labels = data.aisle,rotation=90, fontsize=12)

orders_df.head()
order_products_train_df.head()
order_products_train_df.groupby("add_to_cart_order")["reordered"].aggregate({'reordered_percnt': 'mean'}).sort_values(by="reordered_percnt",ascending= False).reset_index().head(20)
order_products_name_train_df = pd.merge(left=order_products_train_df,right=products_df.loc[:,["product_id","product_name"]],on="product_id",how="left")
common_Products=order_products_name_train_df[order_products_name_train_df.reordered == 1]["product_name"].value_counts().to_frame().reset_index()

plt.figure(figsize=(16,10))

plt.xticks(rotation=90)

sns.barplot(x="product_name", y="index", data=common_Products.head(20))

plt.ylabel('product_name', fontsize=12)

plt.xlabel('count', fontsize=12)

plt.show()
order_products_name_prior_df = pd.merge(left=order_products_prior_df,right=products_df.loc[:,["product_id","product_name"]],on="product_id",how="left")
common_Products_prior=order_products_name_prior_df[order_products_name_prior_df.reordered == 1]["product_name"].value_counts().to_frame().reset_index()

plt.figure(figsize=(16,10))

plt.xticks(rotation=90)

sns.barplot(x="product_name", y="index", data=common_Products_prior.head(20))

plt.ylabel('product_name', fontsize=12)

plt.xlabel('count', fontsize=12)

plt.show()
#Calculating only for the last orders(i.e train data) of each customer to get the idea

order_products_name_train_df = pd.merge(left=order_products_name_train_df,right=products_details.loc[:,["product_id","aisle","department"]],on="product_id",how="left")
common_aisle=order_products_name_train_df["aisle"].value_counts().to_frame().reset_index()

plt.figure(figsize=(16,10))

plt.xticks(rotation=90)

sns.barplot(x="aisle", y="index", data=common_aisle.head(20),palette="Blues_d")

plt.ylabel('aisle', fontsize=12)

plt.xlabel('count', fontsize=12)

plt.show()
common_aisle=order_products_name_train_df["department"].value_counts().to_frame().reset_index()

plt.figure(figsize=(16,10))

plt.xticks(rotation=90)

sns.barplot(x="department", y="index", data=common_aisle,palette="Blues_d")

plt.ylabel('department', fontsize=12)

plt.xlabel('count', fontsize=12)

plt.show()
train_group_reordered = order_products_train_df.groupby(["order_id","reordered"])["product_id"].apply(list).reset_index()
train_group_reordered=train_group_reordered[train_group_reordered.reordered == 1].drop(columns=["reordered"]).reset_index(drop=True)

train_group_reordered.head()
order_products_prior_df.head()
prior_group_reordered = order_products_prior_df.groupby(["order_id","reordered"])["product_id"].apply(list).reset_index()
prior_group_reordered=prior_group_reordered[prior_group_reordered.reordered == 1].drop(columns=["reordered"])

prior_group_reordered=prior_group_reordered.reset_index(drop=True)

prior_group_reordered.head()
User_maxOrder=orders_df.groupby("user_id").order_number.aggregate(np.max).sort_values(ascending= False).reset_index()


User_maxOrder["order_number"].value_counts()

plt.figure(figsize=(16,10))

g=sns.countplot(x="order_number",data=User_maxOrder)

g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(x="order_dow", data=orders_df, color=color[0])

plt.ylabel('Count', fontsize=12)

plt.xlabel('Day of week', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Orders by week day", fontsize=15)

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(x="order_hour_of_day", data=orders_df, color=color[0])

plt.ylabel('Count', fontsize=12)

plt.xlabel('Hour of day', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Orders by Hour of day", fontsize=15)

plt.show()