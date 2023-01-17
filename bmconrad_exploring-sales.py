import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *
items = pd.read_csv("../input/items.csv" )
items_categories =  pd.read_csv("../input/item_categories.csv")
shops =  pd.read_csv("../input/shops.csv")
sales_train =  pd.read_csv("../input/sales_train.csv")
sales_test =  pd.read_csv("../input/test.csv")
example =  pd.read_csv("../input/sample_submission.csv")

items.head()
items_categories.head()
shops.head()

sales_test.head()
df_train = pd.merge(pd.merge(pd.merge(sales_train, items, on="item_id"), items_categories, on="item_category_id"), shops, on="shop_id")
df_train = df_train[["item_id","item_category_id","shop_id","date", "date_block_num","item_name","item_category_name", "shop_name", "item_price", "item_cnt_day"]]

# Change types
def reformat(someDate):
    s = someDate.split(".")
    newDate = s[1] + "/" + s[0] + "/" + s[2]
    return newDate
df_train.date = df_train.date.transform(reformat)
print (df_train.date.head())
df_train.item_id = df_train.item_id.astype('category')
df_train.item_category_id = df_train.item_category_id.astype('category')
df_train.shop_id = df_train.shop_id.astype('category')
df_train.item_name = df_train.item_name.astype('category')
df_train.item_category_name = df_train.item_category_name.astype('category')
df_train.shop_name = df_train.shop_name.astype('category')
df_train.date = pd.to_datetime(df_train.date, format='%m/%d/%Y')
df_train.index = df_train.date
df_train.head()

# Next month ... Predict the # of each item sold from each category
df_test = pd.merge(pd.merge(pd.merge(sales_test, items, on="item_id"), items_categories, on="item_category_id"), shops, on="shop_id")
df_test = df_test[["item_id","item_category_id","shop_id","item_name","item_category_name", "shop_name"]] 
df_test.head()

"""
X_train, X_valid, y_train, y_valid = train_test_split(df_train[list(set(df_train.columns.tolist())-set(["item_cnt_day"]))],
                                                      df_train.item_cnt_day,
                                                      test_size=0.33,
                                                      random_state=42)

X_train = X_train[["item_id","item_category_id","shop_id","date", "date_block_num","item_name","item_category_name", "shop_name", "item_price"]]
X_valid = X_valid[["item_id","item_category_id","shop_id","date", "date_block_num","item_name","item_category_name", "shop_name", "item_price"]]
print (X_train.shape)
print (X_valid.shape)
print (y_train.shape)
print (y_valid.shape)
print (df_train.dtypes)
"""
# What did our #sales look like overall?
ax=df_train["item_cnt_day"].plot(title='Overall Items Sold (Past 2 Years)')
month_history=df_train[["item_cnt_day"]].groupby([df_train.index.month, df_train.index.year]).sum()
month_history["month"]=list(map(lambda x:x[0],month_history.index.values.tolist()))
month_history["year"]=list(map(lambda x:x[1],month_history.index.values.tolist()))
ggplot(aes(x='month',y='item_cnt_day'), data=month_history) + geom_point() + geom_line() + stat_smooth(color='blue', span=0.2) + facet_wrap('year')+ggtitle('AAA')
### Setup data

df_train.set_index('date', inplace=True)
df_train.reset_index(inplace=True)
y = df_train.item_cnt_day.as_matrix()
X=df_train[["date_block_num","item_id","item_category_id","shop_id"]].as_matrix()

## Exponential smoothing for our input
ewma = pandas.stats.moments.ewma
Xexp = ewma(X, com=2)

## Linea Regression Classifier
clf = LinearRegression()
clf_fit = clf.fit ( Xexp, y )
print(clf_fit.coef_)

## Next month prediction as input
date_block=34
df_test["date_block_num"] = 34

## All 
Xt = df_test[["date_block_num","item_id","item_category_id","shop_id"]]
initial_out=clf_fit.predict(Xt)


my_submission=pd.DataFrame({'ID':[i for i in range(len(initial_out))],'item_cnt_day':initial_out})
my_submission.to_csv("out1.csv", sep=',', encoding='utf-8',index=False)
out1=pd.read_csv(os.listdir(os.getcwd())[0])
out1
samp=pd.read_csv("../input/sample_submission.csv")
samp









