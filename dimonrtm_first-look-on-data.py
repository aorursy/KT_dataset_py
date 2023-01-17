# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

from xgboost import XGBRegressor

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import RandomizedSearchCV,cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import make_scorer

from itertools import product

import re

import nltk

from nltk import word_tokenize

from nltk.corpus import stopwords



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
!pip install pymorphy2

!pip install pymorphy2-dicts

!pip install DAWG-Python
import pymorphy2
sample_submissions=pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

sales_train_df=pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

test_df=pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

items_df=pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

shops_df=pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

item_categories_df=pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
sample_submissions.head()
sales_train_df.head()
sales_train_df.info()
test_df.head()
test_df.info()
sales_train_df.describe()
test_df.describe()
print(sales_train_df["shop_id"].nunique())

print(test_df["shop_id"].nunique())
print(sales_train_df["item_id"].nunique())

print(test_df["item_id"].nunique())
plt.figure(figsize=(20,6))

sales_train_df["item_cnt_day"].hist(bins=200)
plt.figure(figsize=(20,6))

sales_train_df["item_price"].hist(bins=300)
plt.figure(figsize=(20,6))

sales_train_df["shop_id"].hist(bins=60)
plt.figure(figsize=(20,6))

test_df["shop_id"].hist(bins=60)
plt.figure(figsize=(20,6))

sales_train_df["item_id"].hist(bins=1000)
plt.figure(figsize=(20,6))

test_df["item_id"].hist(bins=1000)
nltk.download("stopwords")
def text_preprocessing(text):

  txt=""

  morph=pymorphy2.MorphAnalyzer()

  stop_words_ru=stopwords.words("russian")

  stop_words_eng=stopwords.words("english")

  stroka=re.sub('[^A-Za-zА-Яа-я]+',' ',text)

  strs=stroka.split(" ")

  filtered_strs=[morph.parse(w.lower())[0].normal_form for w in strs if (w not in stop_words_ru and w not in stop_words_eng)]

  txt=" ".join(filtered_strs)

  return txt
'''%%time

items_df["item_name"]=items_df["item_name"].apply(lambda x:text_preprocessing(x))'''
items_df["item_name"]
sales_train_df=sales_train_df.join(shops_df.set_index("shop_id"),on="shop_id",how="left").join(

items_df.set_index("item_id"),on="item_id",how="left").join(item_categories_df.set_index("item_category_id")

                                                              ,on="item_category_id",how="left")

sales_train_df.head().T
test_df=test_df.join(shops_df.set_index("shop_id"),on="shop_id",how="left").join(

items_df.set_index("item_id"),on="item_id",how="left").join(item_categories_df.set_index("item_category_id")

                                                              ,on="item_category_id",how="left")

test_df.head().T
plt.figure(figsize=(20,6))

sales_train_df["item_category_id"].hist(bins=84)
plt.figure(figsize=(20,6))

test_df["item_category_id"].hist(bins=84)
sns.relplot(x="item_price",y="item_cnt_day",height=9,aspect=1,data=sales_train_df)
sales_train_df[sales_train_df.item_cnt_day>2000]
sales_train_df[sales_train_df.item_category_id==9]["item_cnt_day"].mean()
sales_train_df[sales_train_df.item_id==11373]["item_cnt_day"].mean()
sales_train_df[(sales_train_df.shop_id==12)&(sales_train_df.item_id==11373)]["item_cnt_day"].mean()
sales_train_df=sales_train_df[~sales_train_df.isin(sales_train_df[sales_train_df.item_cnt_day>2000])]
sales_train_df[sales_train_df.item_price>50000]
sales_train_df=sales_train_df[~sales_train_df.isin(sales_train_df[sales_train_df.item_price>50000])]
sns.relplot(x="item_price",y="item_cnt_day",height=9,aspect=1,data=sales_train_df)
sales_train_df[sales_train_df.item_cnt_day>800]
sales_train_df[sales_train_df.item_id==20949]["item_cnt_day"].mean()
sales_train_df[(sales_train_df.shop_id==12)&(sales_train_df.item_id==20949)]["item_cnt_day"].mean()
sales_train_df=sales_train_df[~sales_train_df.isin(sales_train_df[sales_train_df.item_cnt_day>800])]
sns.relplot(x="item_price",y="item_cnt_day",height=9,aspect=1,data=sales_train_df)
item_id_uniq=pd.unique(sales_train_df["item_id"])
test_unique_items=test_df[~test_df["item_id"].isin(item_id_uniq)]

test_unique_items.head()
test_unique_items.shape
test_unique_items["item_id"].nunique()
sales_train_df.shape
%%time

train=[]

col=["date_block_num","shop_id","item_id"]

for i in range(34):

  sales=sales_train_df[sales_train_df.date_block_num==i]

  train.append(np.array(list(product([i],sales.shop_id.unique(),sales.item_id.unique())),dtype="int16"))

train=pd.DataFrame(np.vstack(train),columns=col)

train["date_block_num"]=train["date_block_num"].astype(np.int8)

train["shop_id"]=train["shop_id"].astype(np.int8)

train["item_id"]=train["item_id"].astype(np.int16)

print(train.head())
month_df=sales_train_df[["date_block_num","shop_id","item_id","item_cnt_day"]].groupby(["date_block_num","shop_id","item_id"]).sum()

month_df.head()
date_block_total=month_df.reset_index()

date_block_total_group=date_block_total[["date_block_num","item_cnt_day"]].groupby("date_block_num",as_index=False).sum()

plt.figure(figsize=(20,6))

sns.barplot(x="date_block_num",y="item_cnt_day",data=date_block_total_group)
date_block_total["year"]=date_block_total.date_block_num.apply(lambda x:((x//12)+2013))

date_block_total["month"]=date_block_total.date_block_num.apply(lambda x: (x%12))

date_block_total.head()
total_month=date_block_total[["month","item_cnt_day"]].groupby("month",as_index=False).sum()

plt.figure(figsize=(10,10))

sns.barplot(x="month",y="item_cnt_day",data=total_month)
total_year=date_block_total[["year","item_cnt_day"]].groupby("year",as_index=False).sum()

plt.figure(figsize=(6,6))

sns.barplot(x="year",y="item_cnt_day",data=total_year)
train["year"]=train.date_block_num.apply(lambda x:((x//12)+2013))

train["month"]=train.date_block_num.apply(lambda x: (x%12))

train["year"]=train["year"].astype(np.int16)

train["month"]=train["month"].astype(np.int8)

train.head()
test_df=pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

test_df["date_block_num"]=34

test_df["year"]=((34//12)+2013)

test_df["month"]=(34%12)

test_df["date_block_num"]=test_df["date_block_num"].astype(np.int8)

test_df["shop_id"]=test_df["shop_id"].astype(np.int8)

test_df["item_id"]=test_df["item_id"].astype(np.int16)

test_df["year"]=test_df["year"].astype(np.int16)

test_df["month"]=test_df["month"].astype(np.int8)

test_df.head()
categorical=["year","month"]
train=train.join(month_df,on=["date_block_num","shop_id","item_id"],how="left")

train["item_cnt_day"]=train["item_cnt_day"].fillna(0).clip(0,20).astype(np.float16)

train.head()
train=pd.concat([train,test_df],ignore_index=True,sort=False,keys=col)

train.fillna(0,inplace=True)

train.head()
def create_lag_features(df,feature):

  numeric_features=[]

  for i in range(1,7):

    lagged=df.copy()

    lagged=lagged[["date_block_num","shop_id","item_id",feature]]

    lagged.columns=["date_block_num","shop_id","item_id",feature+"_lag_"+str(i)]

    numeric_features.append(feature+"_lag_"+str(i))

    lagged["date_block_num"]+=i

    df=df.join(lagged.set_index(["date_block_num","shop_id","item_id"]),on=["date_block_num","shop_id","item_id"],how="left")

  return df,numeric_features
%%time

train,n_col=create_lag_features(train,"item_cnt_day")

train=train.fillna(0)

numeric=n_col
train.tail().T
total_cnt_for_items=sales_train_df[["date_block_num","item_id","item_cnt_day"]].groupby(["date_block_num","item_id"]).sum()

total_cnt_for_items.columns=["total_cnt_for_items"]

total_cnt_for_items.head()
train=train.join(total_cnt_for_items,on=["date_block_num","item_id"],how="left")

train.fillna(0,inplace=True)

train["total_cnt_for_items"]=train["total_cnt_for_items"].astype(np.float16)

train.head().T
%%time

train,n_col=create_lag_features(train,"total_cnt_for_items")

train=train.fillna(0)

train=train.drop(["total_cnt_for_items"],axis=1)

numeric=numeric+n_col
train.tail().T
total_cnt_for_shops=sales_train_df[["date_block_num","shop_id","item_cnt_day"]].groupby(["date_block_num","shop_id"]).sum()

total_cnt_for_shops.columns=["total_cnt_for_shops"]

total_cnt_for_shops.head()
sum_cnt_shops=total_cnt_for_shops.reset_index()

cnt_shops=sum_cnt_shops[["shop_id","total_cnt_for_shops"]].groupby("shop_id",as_index=False).mean()

cnt_shops=cnt_shops.join(shops_df.set_index(["shop_id"]),on="shop_id",how="left")

cnt_shops.head()
plt.figure(figsize=(15,15))

sns.barplot(x="total_cnt_for_shops",y="shop_name",orient="h",data=cnt_shops)
cnt_shops.loc[cnt_shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

cnt_shops['shop_city'] = cnt_shops['shop_name'].str.split(' ').map(lambda x: x[0])

cnt_shops.loc[cnt_shops.shop_city == '!Якутск', 'shop_city'] = 'Якутск'

cnt_shops['shop_type'] = cnt_shops['shop_name'].str.split(' ').map(lambda x: x[1])

cnt_shops.head()

sum_cnt_shop_city=cnt_shops[["shop_city","total_cnt_for_shops"]].groupby("shop_city",as_index=False).sum()

plt.figure(figsize=(15,15))

sns.barplot(x="total_cnt_for_shops",y="shop_city",orient="h",data=sum_cnt_shop_city)
sum_cnt_shop_type=cnt_shops[["shop_type","total_cnt_for_shops"]].groupby("shop_type",as_index=False).mean()

plt.figure(figsize=(15,15))

sns.barplot(x="total_cnt_for_shops",y="shop_type",orient="h",data=sum_cnt_shop_type)
train=train.join(cnt_shops[["shop_id","shop_city","shop_type"]].set_index(["shop_id"]),on="shop_id",how="left")

train.fillna(0,inplace=True)

train["shop_city"]=train["shop_city"].astype(str)

train["shop_type"]=train["shop_type"].astype(str)

train.head().T
categorical=categorical+["shop_city","shop_type"]
train=train.join(total_cnt_for_shops,on=["date_block_num","shop_id"],how="left")

train.fillna(0,inplace=True)

train["total_cnt_for_shops"]=train["total_cnt_for_shops"].astype(np.float16)

train.head().T
%%time

train,n_col=create_lag_features(train,"total_cnt_for_shops")

train=train.fillna(0)

train=train.drop(["total_cnt_for_shops"],axis=1)

numeric=numeric+n_col
train.tail().T
total_cnt_for_item_categories=sales_train_df[["date_block_num","item_category_id","item_cnt_day"]].groupby(["date_block_num","item_category_id"]).sum()

total_cnt_for_item_categories.columns=["total_cnt_for_item_categories"]

total_cnt_for_item_categories.head()
sum_cnt_for_item_category=total_cnt_for_item_categories.reset_index()

sum_cnt_for_item_category=sum_cnt_for_item_category[["item_category_id","total_cnt_for_item_categories"]].groupby("item_category_id",as_index=False).sum()

sum_cnt_for_item_category=sum_cnt_for_item_category.join(item_categories_df.set_index("item_category_id"),on=["item_category_id"],how="left")

plt.figure(figsize=(15,15))

sns.barplot(x="total_cnt_for_item_categories",y="item_category_name",orient="h",data=sum_cnt_for_item_category)
sum_cnt_for_item_category["category"]=sum_cnt_for_item_category.item_category_name.str.split("-").str[0]

sum_cnt_for_item_category["subcategory"]=sum_cnt_for_item_category.item_category_name.str.split("-").str[1]

sum_cnt_for_item_category.head()
sum_cnt_category=sum_cnt_for_item_category[["category","total_cnt_for_item_categories"]].groupby("category",as_index=False).sum()

plt.figure(figsize=(15,15))

sns.barplot(x="total_cnt_for_item_categories",y="category",orient="h",data=sum_cnt_category)
sum_cnt_subcategory=sum_cnt_for_item_category[["subcategory","total_cnt_for_item_categories"]].groupby("subcategory",as_index=False).sum()

plt.figure(figsize=(15,15))

sns.barplot(x="total_cnt_for_item_categories",y="subcategory",orient="h",data=sum_cnt_subcategory)
train=train.join(items_df.set_index(["item_id"]),on=["item_id"],how="left")

train=train.join(sum_cnt_for_item_category[["item_category_id","category","subcategory"]].set_index(["item_category_id"]),on=["item_category_id"],how="left")

train.fillna(0,inplace=True)

train["category"]=train["category"].astype(str)

train["subcategory"]=train["subcategory"].astype(str)

train.tail().T
categorical=categorical+["category","subcategory"]
train=train.join(total_cnt_for_item_categories,on=["date_block_num","item_category_id"],how="left")

train.fillna(0,inplace=True)

train["total_cnt_for_item_categories"]=train["total_cnt_for_item_categories"].astype(np.float16)

train.head().T
%%time

train,n_col=create_lag_features(train,"total_cnt_for_item_categories")

train=train.fillna(0)

train=train.drop(["total_cnt_for_item_categories"],axis=1)

numeric=numeric+n_col
train.tail().T
train["item_name"]
mean_price=sales_train_df[["date_block_num","shop_id","item_id","item_price"]].groupby(["date_block_num","shop_id","item_id"]).mean()

mean_price.columns=["mean_price"]

mean_price.head()
train=train.join(mean_price,on=["date_block_num","shop_id","item_id"],how="left")

train.fillna(0,inplace=True)

train["mean_price"]=train["mean_price"].astype(np.float16)

train.head().T
%%time

train,n_col=create_lag_features(train,"mean_price")

train=train.fillna(0)

train=train.drop(["mean_price"],axis=1)

numeric=numeric+n_col
train.tail().T
mean_price_for_items=sales_train_df[["date_block_num","item_id","item_price"]].groupby(["date_block_num","item_id"]).mean()

mean_price_for_items.columns=["mean_price_for_items"]

mean_price_for_items.head()
train=train.join(mean_price_for_items,on=["date_block_num","item_id"],how="left")

train.fillna(0,inplace=True)

train["mean_price_for_items"]=train["mean_price_for_items"].astype(np.float16)

train.head().T
%%time

train,n_col=create_lag_features(train,"mean_price_for_items")

train=train.fillna(0)

train=train.drop(["mean_price_for_items"],axis=1)

numeric=numeric+n_col
train.tail().T
mean_price_for_item_categories=sales_train_df[["date_block_num","item_category_id","item_price"]].groupby(["date_block_num","item_category_id"]).mean()

mean_price_for_item_categories.columns=["mean_price_for_item_categories"]

mean_price_for_item_categories.head()
train=train.join(mean_price_for_item_categories,on=["date_block_num","item_category_id"],how="left")

train.fillna(0,inplace=True)

train["mean_price_for_item_categories"]=train["mean_price_for_item_categories"].astype(np.float16)

train.head().T
%%time

train,n_col=create_lag_features(train,"mean_price_for_item_categories")

train=train.fillna(0)

train=train.drop(["mean_price_for_item_categories"],axis=1)

numeric=numeric+n_col
train.tail().T
sales_train_df["revenue"]=sales_train_df["item_price"]*sales_train_df["item_cnt_day"]

total_revenue=sales_train_df[["date_block_num","shop_id","revenue"]].groupby(["date_block_num","shop_id"]).sum()

total_revenue.columns=["total_shop_revenue"]

total_revenue.head()
train=train.join(total_revenue,on=["date_block_num","shop_id"],how="left")

train["total_shop_revenue"]=train["total_shop_revenue"].astype(np.float32)

train.fillna(0,inplace=True)

train.head().T
%%time

train,n_col=create_lag_features(train,"total_shop_revenue")

train=train.fillna(0.0)

train=train.drop(["total_shop_revenue"],axis=1)

numeric=numeric+n_col
train.tail().T
test_df=train[train.date_block_num==34]

train=train[train.date_block_num<34]

train=train[~train["date_block_num"].isin([0,1,2,3,4,5])].reset_index(drop=True)

train.head().T
test_df.head().T
groups = train.groupby(train.date_block_num).groups

sorted_groups = [value for (key, value) in sorted(groups.items())]

cv=[(np.concatenate(sorted_groups[:8]),np.concatenate(sorted_groups[8:])),

    (np.concatenate(sorted_groups[:16]),np.concatenate(sorted_groups[16:])),

    (np.concatenate(sorted_groups[:24]),np.concatenate(sorted_groups[24:]))]
y_train=train["item_cnt_day"]
X_train_categorical=train[categorical]

X_test_categorical=test_df[categorical]

X_train_categorical["subcategory"]=X_train_categorical["subcategory"].astype(str)

X_test_categorical["subcategory"]=X_test_categorical["subcategory"].astype(str)

X_test_categorical["year"]=X_test_categorical["year"].astype(np.int16)

X_test_categorical["month"]=X_test_categorical["month"].astype(np.int8)

X_test_categorical.loc[X_test_categorical.category=="PC ","category"]="Игры PC "

X_test_categorical.loc[X_test_categorical.subcategory==" Гарнитуры/Наушники","subcategory"]=" Аксессуары для игр"
for feature in categorical:

  le=LabelEncoder()

  print(feature)

  X_train_categorical[feature]=le.fit_transform(X_train_categorical[feature])

  X_test_categorical[feature]=le.transform(X_test_categorical[feature])
X_train_categorical.head()
X_test_categorical.head()
X_train_numeric=train[numeric]

X_test_numeric=test_df[numeric]
X_train_numeric.head().T
X_test_numeric.head().T
label_cat_not_num_train=pd.concat([X_train_categorical,X_train_numeric],axis=1)

label_cat_not_num_train.head().T
label_cat_not_num_test=pd.concat([X_test_categorical,X_test_numeric],axis=1)

label_cat_not_num_test.head().T
label_cat_not_num_train.shape
def downcast_type(df):

  for feature in categorical:

    df[feature]=df[feature].astype(np.int8)
downcast_type(label_cat_not_num_train)

label_cat_not_num_train.info()
def RMSE(y,predictions):

  return  np.sqrt(mean_squared_error(y,predictions))

scorer=make_scorer(RMSE,False)
%%time 

baseline = -cross_val_score(

    XGBRegressor(max_depth=10,

                 subsample=0.8,

                 colsample_bytree=0.9,

                 colsample_bylevel=0.7,

                 min_child_weight=200,

                 n_estimators=1000,

                 learning_rate=0.025,

                 objective="reg:squarederror",

                 tree_method="hist"), label_cat_not_num_train, y_train, scoring=scorer,cv=cv

).mean()

print(baseline)
%%time

estimator=XGBRegressor(max_depth=10,

                       subsample=0.8,

                       colsample_bytree=0.9,

                       colsample_bylevel=0.7,

                       min_child_weight=200,

                       n_estimators=1000,

                       learning_rate=0.025,

                       objective="reg:squarederror",

                       tree_method="hist")

estimator.fit(label_cat_not_num_train,y_train)

importances=estimator.feature_importances_

predictions=estimator.predict(label_cat_not_num_test)
importances
sample_submissions["item_cnt_month"]=predictions

sample_submissions.head()
sample_submissions.to_csv("xgboost_lagged_features_6.csv",index=False)