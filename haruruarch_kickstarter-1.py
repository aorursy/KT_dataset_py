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
import keras

print(keras.__version__)

from keras.utils.np_utils import to_categorical



import sklearn as sk

print(sk.__version__)



import matplotlib

import matplotlib.pyplot as plt

print(matplotlib.__version__)

%matplotlib inline



import seaborn as sns

print(sns.__version__)
ks_2016 = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201612.csv", header = 0, encoding='macroman')

ks_2016.head()
ks_2018 = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv", header = 0)

ks_2018.head(30)
ks_2018.describe()
#いらない列を消す

ks_2018_modified = ks_2018[["category","main_category", "state", "usd_goal_real", "country", "usd_pledged_real", "deadline", "currency"]]

ks_2018_modified["launched"] = pd.to_datetime([i[:10] for i in ks_2018['launched']])

ks_2018_modified['period'] = pd.to_datetime(ks_2018['deadline']) - ks_2018_modified["launched"] 

ks_2018_modified['period'] = ks_2018_modified['period'] / np.timedelta64(1,'D')

ks_2018_modified["percentage"] = (ks_2018["usd_pledged_real"] / ks_2018["usd_goal_real"]) * 100

ks_2018_modified.describe()
ks_2018_modified.head(30)
countries = ks_2018_modified["country"].unique()

num_countries = len(countries)

states = ks_2018["state"].unique()

categories = ks_2018_modified["main_category"].unique()

num_categories = len(categories)

print(countries)

print(num_countries, "countries")

print(states)

print(categories)

print(num_categories, "categoies")
#目的変数と説明変数の関係を確認するためのグラフを作成する

pd.plotting.scatter_matrix(ks_2018_modified, figsize=(10, 10))

plt.show()
#目的変数を説明するのに有効そうな説明変数を見つける

ks_2018_modified.corr()
sns.heatmap(ks_2018_modified.corr())

plt.show()

#あんまり有効な情報ゲットできず
sns.catplot(x = "country", y = "percentage", data = ks_2018_modified,  height = 15)

plt.show()

#そもそも国ごとに数の偏りがありそう

#アメリカは目標額を大きく上回ったものがある

#ほとんど失敗ぽそうなので、成功例を抜きだしてなぜ成功したのか考えたほうがよさそう(とおもったけど、y軸の範囲とプロットの数のせいかもしれない)
plt.figure(figsize = (15, 10))

v = sns.violinplot(x = "country", y = "percentage", data = ks_2018_modified,  width = 1.5, bw = "silverman")

v.set(ylim = (0, 500))

plt.show()

#上の図はy軸が大成功したプロジェクトに引っ張られて見にくかったので、y軸の範囲を決めた拡大版をつくってみた

#100のところ（成功と失敗の線引き）で傾向がある

#国ごとに傾向が違う、たとえば日本は失敗が多い
ks_2018.groupby("state").count()

#成功しているやつ意外と多かった
c = sns.catplot(x = "state", y = "percentage", data = ks_2018_modified,  height = 15)

c.set(ylim = (0, 500))

plt.show()

#成功していないのに成功になってるやつがある？
sns.distplot([i for i in ks_2018_modified["percentage"] if i >= 100], hist=False, color="g", kde_kws={"shade": True})

#成功しているものの達成率の分布
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.figure(figsize = (15, 10))

s = sns.scatterplot(x = "launched", y = "percentage", data = ks_2018_modified, hue = "currency")

s.set(ylim = (0, 1000))

s.set(xlim = (pd.to_datetime("2009-01-01"), pd.to_datetime("2018-05-01")))

plt.show()

#開始時期によってKickstarterがバズったりすることで成功率が上がっているというような傾向がないかさぐってみたかった、あと通貨による影響があるか

#開始時期がおかしいやつがあったので、時期を指定した

#そもそも年ごとに母数がちがった
ax = sns.jointplot((ks_2018_modified["launched"] - pd.to_datetime("2009-01-01")) / np.timedelta64(1, 'D'), ks_2018_modified["percentage"], xlim = (0, 3500), ylim = (0, 15000), color = "g")



#開始時期（2009-01-01からの日数）と達成率
# ax = sns.kdeplot(ks_2018_modified["usd_goal_real"], ks_2018_modified["percentage"], n_levels = 10,

#                  xlim = (0, 500000), ylim = (0, 25000), cmap="Blues", shade=True, shade_lowest=False)

# #目標金額と達成率、kde重すぎて動かない(;_;)
plt.figure(figsize = (15, 10))

p = sns.scatterplot(x = "period", y = "percentage", data = ks_2018_modified, hue = "main_category")

p.set(ylim = (0, 1000))

p.set(xlim = (0, 100))

plt.show()

#カテゴリーごとの期間と達成率

#下にオレンジが多くて、上に緑が多い？
"""

アルゴリズムを利用する

    回帰の場合は線形回帰、分類の場合はロジスティック回帰

    質的変数が扱えないアルゴリズムを使う場合は、ダミー変数に置き換える

"""

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix





X = pd.get_dummies(ks_2018_modified[["category","main_category", "launched", "usd_goal_real", "country", "deadline", "currency", "period"]], columns = ["category","main_category", "country", "currency"])

X["launched"] = [(pd.to_datetime(i) - pd.to_datetime("2009-01-01")) / np.timedelta64(1, 'D') for i in ks_2018_modified["launched"]]

X["deadline"] = [(pd.to_datetime(i) - pd.to_datetime("2009-01-01")) / np.timedelta64(1, 'D') for i in ks_2018_modified["deadline"]]

X2 = pd.get_dummies(ks_2018_modified[["main_category", "usd_goal_real", "country", "period"]], columns = ["main_category", "country"])

X.head()
#目標達成率（回帰）

y1 = ks_2018_modified["percentage"]

print(y1.head())

lr1 = LinearRegression(fit_intercept = True)

lr1.fit(X, y1)





#成功するかどうか（分類）

y2 = [i == "successful" for i in ks_2018["state"]]

lr2 = LogisticRegression(random_state = 0, verbose = 1).fit(X2, y2) 
"""

予測精度または識別精度を確認する

    回帰問題の場合は、MSE、RMSE、MAEを求める

    分類問題の場合は、混同行列を作成し、Accuracy、Recall、Precisionを求める

"""



#目標達成率（回帰）

y1_estimated = lr1.predict(X)

mse = mean_squared_error(y1, y1_estimated)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y1, y1_estimated)

print("MSE: {}, RMSE: {}, MAE: {}".format(mse, rmse, mae))



#成功するかどうか（分類）

accuracy = lr2.score(X2, y2)

print("Accuracy: ", accuracy)

y2_estimated = lr2.predict(X2)

c_matrix = pd.DataFrame(confusion_matrix(y2, y2_estimated), 

                        index=['正解 = 成功', '正解 = その他'], 

                        columns=['予測 = 成功', '予測 = その他'])

#Recall（実際に正しいもののうち、正であると予測された割合）

recall = c_matrix["予測 = 成功"][0] / (c_matrix["予測 = 成功"][0] + c_matrix["予測 = その他"][0])

#Precision（正と予測したもののうち、どれくらい正しかったか）

precision = c_matrix["予測 = 成功"][0] / (c_matrix["予測 = 成功"][0] + c_matrix["予測 = 成功"][1]) 

print("Recall: {}, Precioin: {}".format(recall, precision))

c_matrix