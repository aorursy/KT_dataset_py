# -*- coding: utf-8 -*-

#ライブラリの読み込み

import pandas as pd

from IPython.display import display

from dateutil.parser import parse

import matplotlib.pyplot as plt
#データの読み込み

df_data = pd.read_csv("../input/kc_house_data.csv")



print("")

print("データセットの頭出し")

display(df_data.head())
# date列の変換（日付の形に変更） (説明変数として使わないため、実行しない。)

#df_data["date"] = [ parse(i[:-7]).date() for i in df_data["date"]]

#display(df_data.head())
# 欠損値のデータが含まれているかどうか確認する

pd.DataFrame(df_data.isnull().sum(), columns=["num of missing"])
#不要な列の削除

df_data_main = df_data.drop(["id","date","zipcode"], axis=1)

df1 = df_data_main.iloc[:,:9]

display(df1.head())

df2 = df_data_main.iloc[:,[0]+list(range(9,18))]

display(df2.head())
# describe（記述統計量の算出）

df_data.describe()
# 散布図行列

pd.plotting.scatter_matrix(df1,figsize=(10,10))

plt.show()

pd.plotting.scatter_matrix(df2,figsize=(10,10))

plt.show()
import itertools

li_combi = list(itertools.combinations(df_data_main.columns[0:], 2))

for X,Y in li_combi:

    if X=='price':

        print("X=%s"%X,"Y=%s"%Y)

        df_data_main.plot(kind="scatter",x=X,y=Y,alpha=0.7,s=10,c="price",colormap="winter")#散布図の作成

        plt.xlabel(X)

        plt.ylabel(Y)

        plt.tight_layout()

        plt.show()#グラフをここで描画させるための行
df_data_main.corr()
for col in df_data_main.columns:

    print(col)

    df_data_main[col].hist()

    plt.xlabel(col)

    plt.ylabel("num")

    plt.show()
for col in df_data_main.columns:

    print(col)

    df_data_main.boxplot(column=col)

    plt.xlabel(col)

    plt.ylabel("num")

    plt.show()
#異常値を除外したグラフを描画する。

for col in df_data_main.columns:

    

    Q1 = df_data_main[col].quantile(.25)

    Q3 = df_data_main[col].quantile(.75)



    #print(Q1)

    #print(Q3)

    

    IQR = Q3 - Q1

    threshold = Q3 + 1.5*IQR



    df_outlier = df_data_main[(df_data_main[col] < threshold)]



    print(col)

    df_outlier.boxplot(column=col)

    plt.xlabel(col)

    plt.ylabel("num")

    plt.show()