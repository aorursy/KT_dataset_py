# Kaggle internals
import os
print(os.listdir("../input"))

# Libraries
from math import sqrt
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Data
test=pd.read_csv('../input/competitive-data-science-final-project/test.csv.gz',compression='gzip')
sales_train=pd.read_csv("../input/coursera-competition-getting-started-eda/sales_train.csv")               # From my EDA Kernel
# Data cleaning
# I need to have clean data that I can use to train 

# No NaNs in the data
# Shops are cleaned up
# Should I do something with negative item counts? Not for now, I think its ok.
# Probably cleaning the outliers - the B2B deal values could be a good idea. But lets leave it for now.
# I have to aggregate the figures to monthly values, right?
# Construct a feature into monthly aggregated figures with "working days"

# The value to be found (y) is "item_cnt_month"
# IDs are categorical features. Do I have to encode item_ids eg one_hot? Huge and sparse!
"""
date: We dont need
date_block_num: keep 
shop_id: keep
item_id: keep
item_price: keep and construct groups
day: delete
month: keep (for seasonality)
year: delete
weekday: delete
item_category_id: keep
items_english: delete
meta_category: keep
shops_english: delete
town: keep
region: keep
revenue: delete
value: delete
"""

# Aggregate to monthly counts
df=pd.DataFrame()
df=sales_train.groupby(["date_block_num","shop_id","item_id","month","item_price","item_category_id","meta_category","town","region"],as_index=False).sum()[["date_block_num","shop_id","item_id","month","item_price","item_category_id","meta_category","town","region","item_cnt_day"]]
df["item_cnt_day"].clip(0,20,inplace=True)
df=df.rename(columns = {'item_cnt_day':'item_cnt_month'})
# unravel
df.head()
# Feature selection
# Features are the input variables to the model
"""
date_block_num: numerical 
shop_id: categorical
item_id: categorical
item_price: numerical
month: categorical
item_category_id: categorical
meta_category: categorical
town: categorical
region: categorical
"""

# Create price categories
df["price_category"]=np.nan
df["price_category"][(df["item_price"]>=0)&(df["item_price"]<=100)]=0
df["price_category"][(df["item_price"]>100)&(df["item_price"]<=200)]=1
df["price_category"][(df["item_price"]>200)&(df["item_price"]<=400)]=2
df["price_category"][(df["item_price"]>400)&(df["item_price"]<=750)]=3
df["price_category"][(df["item_price"]>750)&(df["item_price"]<=1000)]=4
df["price_category"][(df["item_price"]>1000)&(df["item_price"]<=2000)]=5
df["price_category"][(df["item_price"]>2000)&(df["item_price"]<=3000)]=6
df["price_category"][df["item_price"]>3000]=7

sns.countplot(df.price_category)

# Label encode meta_category, town and region
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(df.meta_category)
df["meta_category"]=le.transform(df.meta_category)

le.fit(df.town)
df["town"]=le.transform(df.town)

le.fit(df.region)
df["region"]=le.transform(df.region)

df.head()
X_train=df.drop("item_cnt_month", axis=1)
# Reason for dropping item_price explained below
y_train=df["item_cnt_month"]

X_train.fillna(0, inplace=True)
linmodel=LinearRegression()
linmodel.fit(X_train,y_train)

predictions=linmodel.predict(X_train)
print(sqrt(mean_squared_error(y_train,predictions)))
# Dropping item_price didnt result in a lower RMSE
linmodel.score(X_train, y_train)
# R^2 of 1,6%?! wow.
linmodel.coef_
X_train.columns
# Apply model to test-data

# At first I have to "blow-up" the test data to have some features

# test2=test.merge(X_train[["item_id","shop_id","item_category_id","meta_category","town","region","price_category"]])
# Here several things went bad:
# Merge ignored when e.g. there was no item 5320 in shop 5 and ignored these lines
# If shop,item combinations appeared more than once all lines were added, therefore blowing up the test set

# Following carefully instructions from documentation:
# https://pandas.pydata.org/pandas-docs/stable/merging.html

# And do it step by step. I found out that nearly 40% of item-shop-id pairs are new, therefore lots of nans
# after merge for eg price if i merge on these two items.
# but town, region, category etc depends only on one of them

# Start with adding shop-related values
test2 = pd.merge(test,X_train[["shop_id","town","region"]].drop_duplicates(), how="left",on=["shop_id"])

# Continue with item-related values
test3 = pd.merge(test2,X_train[["item_id","item_category_id","meta_category"]].drop_duplicates(), how="left",on=["item_id"])

# And now the tricky part: The price
# For existing shop-item pairs probably smart to use the latest price point
# As there are different steps to fill the price column I do it in single pieces and later join all methods back together
part1 = pd.merge(test3,X_train[["item_id","shop_id","item_price"]][X_train["date_block_num"]==33].drop_duplicates().groupby(["item_id","shop_id"],as_index=False).max(), how="left",on=["shop_id","item_id"]).dropna()
part2 = pd.merge(test3,X_train[["item_id","shop_id","item_price"]][X_train["date_block_num"]==32].drop_duplicates(), how="left",on=["shop_id","item_id"]).dropna()
part3 = pd.merge(test3,X_train[["item_id","shop_id","item_price"]][X_train["date_block_num"]==31].drop_duplicates(), how="left",on=["shop_id","item_id"]).dropna()
part4 = pd.merge(test3,X_train[["item_id","shop_id","item_price"]][X_train["date_block_num"]==30].drop_duplicates(), how="left",on=["shop_id","item_id"]).dropna()
part5 = pd.merge(test3,X_train[["item_id","shop_id","item_price"]][X_train["date_block_num"]==29].drop_duplicates(), how="left",on=["shop_id","item_id"]).dropna()
part6 = pd.merge(test3,X_train[["item_id","shop_id","item_price"]][X_train["date_block_num"]==28].drop_duplicates(), how="left",on=["shop_id","item_id"]).dropna()
# half a year should be enough

# For non-existing pairs lets use the average price of items
part7 = pd.merge(test3,X_train[["item_id","item_price"]].drop_duplicates().groupby("item_id").mean(), how="left",on=["item_id"]).dropna()

# Now join together. Tricky. Only one value needed. Strict order. Part 1 better than part2, i.e. if 
# part 1 exists dont add part 2 and so forth
prices=part1
print("After adding month -1:")
print(prices.shape)
ids=prices.item_id
prices=pd.concat([prices,part2[~part2.item_id.isin(ids)]])
print("After adding month -2:")
print(prices.shape)
ids=prices.item_id
prices=pd.concat([prices,part3[~part3.item_id.isin(ids)]])
print("After adding month -3:")
print(prices.shape)
ids=prices.item_id
prices=pd.concat([prices,part4[~part4.item_id.isin(ids)]])
print("After adding month -4:")
print(prices.shape)
ids=prices.item_id
prices=pd.concat([prices,part5[~part5.item_id.isin(ids)]])
print("After adding month -5:")
print(prices.shape)
ids=prices.item_id
prices=pd.concat([prices,part6[~part6.item_id.isin(ids)]])
ids=prices.item_id
print("After adding month -6:")
print(prices.shape)

# Until here it is clear. I build up a list of shop-item-pairs and the according prices that exist
# Step 7 is different. This is supposed to fill "the rest" except 15k rows of unknown item_id

# It is not about item_id, it is the shift from shop-item pairs to only according to item price-setting
# I need shop-item-pair IDs
prices["shop_item_pair_id"]=prices.shop_id.astype(str)+"-"+prices.item_id.astype(str)
part7["shop_item_pair_id"]=part7.shop_id.astype(str)+"-"+part7.item_id.astype(str)
prices=pd.concat([prices,part7[~part7.shop_item_pair_id.isin(prices.shop_item_pair_id)]])
prices.drop("shop_item_pair_id", axis=1,inplace=True)
print("After adding prices only according to item_id:")
print(prices.shape)

# Merge with original DF to include also the 15k lines with unknown items
final_test=pd.merge(test3, prices[["shop_id","item_id","item_price"]], how="left",on=["shop_id","item_id"])
final_test=final_test.groupby(["ID"],as_index=False).max()

# Lets add the month and price_category
final_test["date_block_num"]=34
final_test["month"]=11
# This check helped me to find duplicates

counts = final_test['ID'].value_counts()
final_test[final_test['ID'].isin(counts.index[counts > 1])]
# Now we have the issue with NaNs for shop-item pairs that did not exists in X_train
print(final_test[final_test["item_price"].isnull()].head())
# 15k lines of new items. Fill with global price median
final_test.item_price.fillna(final_test.item_price.median(),inplace=True)
final_test.item_category_id.fillna(final_test.item_category_id.median(),inplace=True)
final_test.meta_category.fillna(final_test.meta_category.median(),inplace=True)
print(final_test[final_test["item_price"].isnull()])
print(final_test.shape)

final_test.head(20)
print(X_train.columns)
print(final_test.columns)
# Create price categories
final_test["price_category"]=np.nan
final_test["price_category"][(final_test["item_price"]>=0)&(final_test["item_price"]<=100)]=0
final_test["price_category"][(final_test["item_price"]>100)&(final_test["item_price"]<=200)]=1
final_test["price_category"][(final_test["item_price"]>200)&(final_test["item_price"]<=400)]=2
final_test["price_category"][(final_test["item_price"]>400)&(final_test["item_price"]<=750)]=3
final_test["price_category"][(final_test["item_price"]>750)&(final_test["item_price"]<=1000)]=4
final_test["price_category"][(final_test["item_price"]>1000)&(final_test["item_price"]<=2000)]=5
final_test["price_category"][(final_test["item_price"]>2000)&(final_test["item_price"]<=3000)]=6
final_test["price_category"][final_test["item_price"]>3000]=7
# Right order
final_test=final_test[['ID','date_block_num','shop_id','item_id','month','item_price','item_category_id','meta_category','town','region','price_category']]

# Now predict
predictions=linmodel.predict(final_test.drop("ID",axis=1))
print(predictions)
output=pd.DataFrame()
output["ID"]=final_test["ID"]
output["item_cnt_month"]=predictions
output.head(20)
output.to_csv('linmodel1.csv',index=False)
# Score of 1.99739. Not surprisingly very bad
# Checks to do:
# - Why is amount always between 1 and 2,5? Very low, isnt it?
# - Is is correct that there are so few fitting shop/item-id pairs from the past 6 month? only 30k?
# Still continue with a more meaningful linear model:
## Pivot by month to wide format
train = df.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_month',aggfunc='sum').fillna(0.0)
train.head()
train=train.reset_index()
train.head()
final_train=train.merge(df[["shop_id","item_id","item_price","item_category_id","meta_category","town","region","price_category"]])
final_train.head()
final_train[final_train.shop_id.isnull()]
print(final_train.shape)
final_train.dropna(inplace=True)
print(final_train.shape)
linmodel=LinearRegression()
linmodel.fit(final_train.drop(33,axis=1),final_train[33])

predictions=linmodel.predict(final_train.drop(33,axis=1))

print(sqrt(mean_squared_error(final_train[33],predictions)))
# vs. 2,35 in the naive linear model
linmodel.score(final_train.drop(33,axis=1),final_train[33])
# improved from 1,6% to 89%?! wow.
# The columns that I need in my testset
print(final_train.columns)
print(test.head())
# Now prepare the test-set
# After many many many,... many (really many!!!) tries to fill the panda dataframe I figured out my approach:
# It took me hours of painful tries to pandas-merge the complete df...

# Fill complete set with the roughest estimation (dont use shop_id and item_id at all)
# Increasingly overwrite data with increasing level of knowledge (use item_id only, then overwrite with shop-item_id pairs)
# In this way there will be no NaNs after the process as in step 1 everything is filled and only overwritten when a better estimate/data is available

# Structure:
# Step 1: Delete "ID"
# Step 2: Fill geographic columns (town, region)
# Step 3: Fill 0-33 with median values
# Step 4: Overwrite 0-33 with values where shop, item-pairs are known
# Step 5: Overwrite 0-33 with values where item is known
# Step 6: Fill item_price
# Step 7: Fill price_category
# Step 8: Fill 'item_category_id' and 'meta_category' with median value
# Step 9: Overwrite both, if known item

# Step1:
test.drop("ID",axis=1,inplace=True)
test
# Step 2: Fill geographic columns (town, region)
test.merge(final_train[["shop_id","town","region"]])
test

# Very easily filled are the geographic columns:
unique_shops=final_train.shop_id.unique()
print(unique_shops)

#test.merge(final_train[["shop_id","town","region"]].unique())

# Step 1: Fill with median values for every column


#test2=pd.merge(test,final_train, how="left", on=["shop_id","item_id"])
#test2=test2.groupby("ID",as_index=False).max()

for i in range(33):
    test[i]=final_train[i+1].mean()

test
test2[test2[0].isnull()].head()
print(final_train.groupby("item_id",as_index=False).median().shape)
print(final_train.groupby("item_id",as_index=False).median().head())
final_train.groupby("item_id",as_index=False).median().set_index('item_id')
test2.set_index("item_id",inplace=True)
test2
print(test2.set_index("item_id").fillna(final_train.groupby("item_id",as_index=False).median(),df.set_index('item_id')).head())
# Fill based only on item_id median in month
#print(test2.shape)
#print(test2[test2[0].isnull()])
test2.fillna()
#test2[test2[0].isnull()]=pd.merge(test2[test2[0].isnull()],final_train.groupby("item_id",as_index=False).median(),how="left",on=["item_id"])
#print(pd.merge(test2[test2[0].isnull()],final_train.groupby("item_id",as_index=False).median(),how="left",on=["item_id"]))
#print(test2.shape)
test2

# Shift by one month
# Already better but of course categories, towns need to be one_hot_encoded
# Much more logical will be to add the past months as features
X_train.head()

