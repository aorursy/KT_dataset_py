%matplotlib inline 

#Notebook出力のおまじない



import numpy as np #高速な数値計算

import pandas as pd #データ整備・フレーム操作

import matplotlib.pyplot as plt #データ可視化

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn import linear_model

#scikit-learn＝統計・機械学習



import os #ファイル・ディレクトリ操作

print(os.listdir("../input"))



df_data = pd.read_csv("../input/ks-projects-201801.csv")
df_data.head()
df_data.describe()
df_data.shape

df_data.info()
print(df_data['state'].unique()) #重複要素を除いた場合
plt.scatter(df_data['backers'],df_data['usd pledged'])

plt.xlabel("backers")

plt.ylabel("usd pledged")
scsflg = df_data['state'] == "successful"

df_scs = df_data[scsflg]

failflg = df_data['state'] != "successful"

df_fail = df_data[failflg]
df_scs['backers'].head() #このままだとdfなのでヒストグラムにできない

print(df_scs['backers'].values) #が、pandasで.valuesつけるとarrayに変換可能
plt.hist(df_scs['backers'].values, bins=10) #以上に大きい値があるから表示が一本に見える
plt.hist([x for x in df_scs['backers'].values if x<200], alpha=0.3, color='r', bins=100)

plt.hist([x for x in df_fail['backers'].values if x<200], alpha=0.3,color='b',bins=100)

plt.xlabel("backers")

plt.ylabel("fleq")
plt.hist([x for x in df_scs['pledged'].values if x<1000], alpha=0.3, color='r', bins=100)

plt.hist([x for x in df_fail['pledged'].values if x<1000], alpha=0.3,color='b',bins=100)

plt.xlabel("pledged")

plt.ylabel("fleq")
print(df_data['main_category'].unique()) #重複要素を除いた場合
sns.pairplot(data=df_scs)