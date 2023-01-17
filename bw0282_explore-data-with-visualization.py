
%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


train =pd.read_csv("../input/train.tsv",sep="\t",index_col ='train_id')
test =pd.read_csv("../input/test.tsv",sep="\t",index_col ='test_id')
train
print("amount of train data = {} | amount of test data = {}" .format(len(train), len(test)))

train.head()
test.head()
# check unique data
train_columns = train.columns
for i in train_columns:
    print("{} = {} unique_data".format(i,len(train[i].unique())))
print(len(train))
test_columns = test.columns
for i in test_columns:
    print("{} = {} unique_data".format(i,len(test[i].unique())))
print(len(test))
train.info()
train_columns = train.columns
for i in train_columns:
    print("{0} = {1:.2f}% null_data".format(i,(len(train[train[i].isnull()])/ len(train))*100 ))
test.info()
test_columns = test.columns
for i in test_columns:
    print("{0} = {1:.2f}% null_data".format(i,(len(test[test[i].isnull()])/ len(test))*100 ))
train.head(10)
train.loc[train["brand_name"].isnull(),"brand_name"] ="No_Brand"
test.loc[test["brand_name"].isnull(),"brand_name"] ="No_Brand"
x_train = train.drop("price", axis =1)
y_train = train["price"]
train.describe().astype("float16")
#outlier exist
plt.scatter(train["price"].values,train["price"].index)
grouped = train.groupby("item_condition_id")["price"].aggregate({"count_of_price":'count'}).reset_index()
grouped
count_price = grouped["count_of_price"]
grouped = train.groupby("item_condition_id")["price"].aggregate({"sum_of_price":'sum'}).reset_index()
grouped["standard_of_price"] = grouped["sum_of_price"] / count_price
grouped
grouped["count_of_price"] = count_price
grouped
figure, (axe1,axe2,axe3) =plt.subplots(nrows =3, ncols =1)
figure.set_size_inches(12,10)
sns.barplot(grouped["item_condition_id"], grouped["sum_of_price"], ax = axe1)
sns.barplot(grouped["item_condition_id"], grouped["standard_of_price"], ax = axe2)
sns.barplot(grouped["item_condition_id"], grouped["count_of_price"], ax = axe3)
grouped = train.groupby("brand_name")["price"].aggregate({"sum_of_price":"sum"}).reset_index()
x = train.groupby("brand_name")["price"].aggregate({"count_of_brand":"count"}).reset_index()
grouped["count_of_brand"] = x["count_of_brand"]
top_10_sales = grouped.sort_values("sum_of_price",ascending=False).head(10)
top_10_sales
figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(10,8)
sns.barplot(top_10_sales["brand_name"], top_10_sales["sum_of_price"])
top_10_volume_sales = grouped.sort_values("count_of_brand",ascending=False).head(10)
figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(10,8)
sns.barplot(top_10_volume_sales["brand_name"], top_10_volume_sales["count_of_brand"])
grouped["standard_of_price"] = grouped["sum_of_price"] / grouped["count_of_brand"]
brand_std = grouped.sort_values("standard_of_price",ascending=False)
Top_20_brand_price = brand_std.head(20)
Top_20_brand_price.head()
Top_20_brand_price
figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(35,10)
sns.barplot(Top_20_brand_price["brand_name"], Top_20_brand_price["standard_of_price"])
# grouped = 
grouped = train.groupby("name")["price"].aggregate({"sum_of_price":"sum"}).reset_index()
grouped.head()
grouped["amount_of_product"]= train.groupby("name")["price"].aggregate({"amount_of_product":"count"}).values
grouped["mean_price"] = grouped["sum_of_price"] / grouped["amount_of_product"]
Top_amount_of_product = grouped.sort_values("amount_of_product",ascending=False).head(10)
Top_amount_of_product
train[train["name"]=="BUNDLE"]
figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(20,5)
sns.barplot(Top_amount_of_product["name"], Top_amount_of_product["amount_of_product"])
Top_sales_of_product = grouped.sort_values("sum_of_price",ascending=False).head(10)
Top_sales_of_product
figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(20,5)
sns.barplot(Top_sales_of_product["name"], Top_sales_of_product["sum_of_price"])
Top_mean_price = grouped.sort_values("mean_price",ascending=False).head(30)
Top_mean_price
figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(40,5)
sns.barplot(Top_mean_price["name"], Top_mean_price["mean_price"])
train.head()
grouped = train.groupby("category_name")["price"].aggregate({"sum_of_price":"sum"}).reset_index()
grouped.head()
grouped["amount_of_category"] = train.groupby("category_name").size().values
grouped.head()
grouped["standard_price"] = grouped["sum_of_price"] / grouped["amount_of_category"].astype("float16")
grouped.head()
Top10_sales= grouped.sort_values("sum_of_price",ascending=False).head(10)
Top10_sales
figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(50,8)
sns.barplot(Top10_sales["category_name"],Top10_sales["sum_of_price"])
Top10_amount_category= grouped.sort_values("amount_of_category",ascending=False).head(10)
Top10_amount_category
sample  = train[(train["category_name"] == "Women/Athletic Apparel/Pants, Tights, Leggings") &(train["brand_name"] == "LuLaRoe")
      &(train["item_condition_id"]==1)&(train["shipping"]==1)]
sample.sort_values("price", ascending=False)
figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(50,8)
sns.barplot(Top10_amount_category["category_name"],Top10_amount_category["amount_of_category"])
Top10_mean_price= grouped.sort_values("standard_price",ascending=False).head(10)
Top10_mean_price
figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(50,8)
sns.barplot(Top10_mean_price["category_name"],Top10_mean_price["standard_price"])
train["category_name"] ="Home/Home Appliances/Air Conditioners"
train.head()
train.loc[train["category_name"].isnull(),"category_name"] ="No_category"
test.loc[test["category_name"].isnull(),"category_name"] ="No_category"
train[train["category_name"] =='Home/Home Appliances/Air Conditioners']
grouped = train.groupby("shipping")["price"].aggregate({"sum_of_price":"sum"}).reset_index()
x = train.groupby("shipping")["price"].aggregate({"amount_of_price":"count"}).reset_index()
grouped["amount_of_price"] = x["amount_of_price"]
grouped["mean_of_price"] = grouped["sum_of_price"] / grouped["amount_of_price"]
grouped
figure, (axe1,axe2,axe3) = plt.subplots(nrows =1, ncols =3)
figure.set_size_inches(14,4)
sns.barplot(grouped["shipping"],grouped["amount_of_price"],ax = axe1)
sns.barplot(grouped["shipping"],grouped["sum_of_price"], ax = axe2)
sns.barplot(grouped["shipping"],grouped["mean_of_price"],ax = axe3)