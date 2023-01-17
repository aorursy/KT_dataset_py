# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
print(os.listdir("../input"))
train_raw = pd.read_csv("../input/train.tsv", delimiter='\t')
test_raw = pd.read_csv("../input/test.tsv", delimiter='\t')
train_data = train_raw.head(100000)
# データを見てみる
train_data.head(10)
# 欠損値
def kesson_table(df): 
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : '欠損数', 1 : '%'})
        return kesson_table_ren_columns
kesson_table(train_raw)
kesson_table(test_raw)
# ヒストグラム
def show_dist(df,column_name):
    sns.displot(df[column_name], kde=False)
show_dist(train_raw,"price")
# 40＄以下の商品だけでヒストグラム作成
sns.distplot(train_data[train_data["price"] <= 40.0].price, kde=True)
# 商品状態のヒストグラム
sns.distplot(train_data.item_condition_id, kde=False)
print("max price")
print(train_raw.price.max())
print("min price")
print(train_raw.price.min())
# 商品価格が０円の数
len(train_data[train_data.price == 0])
sns.jointplot("item_condition_id","price", data=train_data[train_data["price"] <= 40.0], kind="reg")
# ユニークな数をカウント
train_raw.brand_name.value_counts(dropna=False)
# データ分析用モデルの作成
train_model = train_raw
train_model.head(3)
# 文字列のカラムをID化する
def string_column_to_int(df,column_name,additional_column_name):
    column_dict = df[column_name].value_counts().to_dict()
    column_map = {}
    index = 0
    for column in column_dict:
        index += 1
        column_map[column] = index
    df[additional_column_name] = df[column_name].map(column_map)
    df[additional_column_name].fillna(0, inplace=True)
    df[additional_column_name] = df[additional_column_name].astype(np.int64)
    return df
train_model = string_column_to_int(train_model,"brand_name","brand_id")
train_model.head(5)
sns.jointplot("brand_id","price",data=train_model, kind="reg")
# category_nameをcategory_1,category_2,category_3に分ける
df = train_model["category_name"].str.split("/",expand=True)
df.rename(columns={0: "category_1",1:"category_2",2:"category_3",3:"category_4",4:"category_5",5:"category_6"},inplace=True)
df.head(5)
# 分割したcategory_nameを結合する
train_model = pd.concat([train_model,df],axis=1)
train_model.head(5)
# categoryそれぞれのユニークな数をカウントする
train_model.category_1.value_counts()
train_model.category_2.value_counts()
train_model.category_3.value_counts()
train_model.category_4.value_counts()
train_model.category_5.value_counts()
train_model = string_column_to_int(train_model,"category_1","category_1")
train_model = string_column_to_int(train_model,"category_2","category_2")
train_model = string_column_to_int(train_model,"category_3","category_3")
train_model = string_column_to_int(train_model,"category_4","category_4")
train_model = string_column_to_int(train_model,"category_5","category_5")
train_model.head(5)
sns.pairplot(train_model[["price","brand_id","category_1","category_2","category_3","category_4","category_5"]])

from sklearn import linear_model
clf = linear_model.LinearRegression()
# 説明変数にitem_condition_idを使用
X = train_model.loc[:,["item_condition_id"]].values
X
# 目的変数にpriceを使用
Y = train_model["price"].values
Y
# 予測モデルを作成 (単回帰)
clf.fit(X, Y)
# 回帰係数と切片の抽出
[a] = clf.coef_
b = clf.intercept_
# 回帰係数
print("回帰係数: ", a)
print("切片: ", b)
print("決定係数: ", clf.score(X,Y))
# 推測する
# 方法→ 目的変数 = 回帰係数 * 説明変数 + 切片
#            → price = a * item_condition_id + b
# matplotlib パッケージを読み込み
import matplotlib.pyplot as plt

# 散布図
#plt.scatter(X,Y)

# 回帰直線
#plt.plot(X,clf.predict(X))
test_raw_X = test_raw.loc[:,["item_condition_id"]].values
test_raw["price"] = clf.predict(test_data_X)
test_raw
# 答え合わせ
submission = pd.read_csv("../input/sample_submission.csv")
submission["price"] = test_raw["price"]
submission.to_csv("submission.csv")
