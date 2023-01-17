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



import sklearn as sk

print(sk.__version__)



import matplotlib

import matplotlib.pyplot as plt

print(matplotlib.__version__)

%matplotlib inline



import seaborn as sns

print(sns.__version__)
ks_2018 = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv", header = 0)

ks_2018.head(10)
ks_2018.describe()
#いらない行を消す

indices = [i == "failed" or i == "successful" for i in ks_2018.state]

ks_2018 = ks_2018[indices]

#いらない列を消す

ks_2018_modified = ks_2018[["category", "state", "usd_goal_real", "country", "usd_pledged_real", "deadline", "currency"]]

ks_2018_modified["launched"] = pd.to_datetime(ks_2018['launched']).dt.date

ks_2018_modified['period'] = pd.to_datetime(ks_2018['deadline']).dt.date - ks_2018_modified["launched"] 

ks_2018_modified['period'] = ks_2018_modified['period'] / np.timedelta64(1,'D')

ks_2018_modified["percentage"] = (ks_2018["usd_pledged_real"] / ks_2018["usd_goal_real"]) * 100

ks_2018_modified["launched_d"] = [(pd.to_datetime(i) - pd.to_datetime("2009-01-01")) / np.timedelta64(1, 'D') for i in ks_2018_modified["launched"]]

ks_2018_modified["deadline_d"] = [(pd.to_datetime(i) - pd.to_datetime("2009-01-01")) / np.timedelta64(1, 'D') for i in ks_2018_modified["deadline"]]

#countryとcurrencyを合体

ks_2018_modified["currency_country"] = list(zip(ks_2018_modified.currency, ks_2018_modified.country))

ks_2018_modified.describe()
ks_2018_modified.head()
ks_2018_modified.describe()
print(ks_2018_modified.isnull().any())

print(ks_2018_modified.isnull().sum())
states = ks_2018["state"].unique()

categories = ks_2018_modified["category"].unique()

num_categories = len(categories)

print(states)

print(categories)

print(num_categories, "categoies")
#目的変数と説明変数の関係を確認するためのグラフを作成する

pd.plotting.scatter_matrix(ks_2018_modified, figsize=(10, 10))

plt.show()
#目的変数を説明するのに有効そうな説明変数を見つける

ks_2018_modified.corr()
sns.catplot(x = "country", y = "percentage", data = ks_2018_modified,  height = 10)

plt.show()
plt.figure(figsize = (15, 5))

v = sns.violinplot(x = "country", y = "percentage", data = ks_2018_modified,  width = 1.5, bw = "silverman")

v.set(ylim = (0, 400))

plt.show()



#国ごとに傾向が違う、たとえば日本は失敗が多い
sns.distplot([i for i in ks_2018_modified["percentage"] if i >= 100], hist=False, color="g", kde_kws={"shade": True})

#成功しているものの達成率の分布
# from pandas.plotting import register_matplotlib_converters

# register_matplotlib_converters()

# plt.figure(figsize = (15, 10))

# s = sns.distplot(ks_2018_modified["launched_d"], y = "percentage", data = ks_2018_modified, hue = "currency")

# plt.show()

# #開始時期によってKickstarterがバズったりすることで成功率が上がっているというような傾向がないかさぐってみたかった、あと通貨による影響があるか

# #開始時期がおかしいやつがあったので、時期を指定した
# ax = sns.jointplot((pd.to_datetime(ks_2018_modified["launched"] - pd.to_datetime("2009-01-01")).dt.date, ks_2018_modified["percentage"], xlim = (0, 3500), ylim = (0, 15000), color = "g")



# #開始時期（2009-01-01からの日数）と達成率
# ax = sns.kdeplot(ks_2018_modified["usd_goal_real"], ks_2018_modified["percentage"], n_levels = 10,

#                  xlim = (0, 500000), ylim = (0, 25000), cmap="Blues", shade=True, shade_lowest=False)

# #目標金額と達成率、kde重すぎて動かない(;_;)



# import itertools

# li_combi = list(itertools.combinations(df_wine.columns[1:], 2))

# for X,Y in li_combi:

#     print("X=%s"%X,"Y=%s"%Y)

#     df_wine.plot(kind="scatter",x=X,y=Y,alpha=0.7,s=10,c="Class label",colormap="winter")#散布図の作成

#     plt.xlabel(X)

#     plt.ylabel(Y)

#     plt.tight_layout()

#     plt.show()#グラフをここで描画させるための行
# # categoryごとのstateの出現頻度を確認

# # データ内のcategoryを抽出しcategoryに格納

# category=df_cloudfound_sct.groupby('category')

# # stateを相対的な頻度に変換

# category=category['state'].value_counts(normalize=True).unstack() 

# # successfulの降順ソート

# category=category.sort_values(by=['successful'],ascending=False)

# # 縦棒グラフ（積み上げ）でグラフ作成

# category[['successful','failed','canceled','live','suspended','undefined']].plot(kind='bar',stacked=True,figsize=(20,20))
# plt.figure(figsize = (15, 10))

# p = sns.scatterplot(x = "period", y = "percentage", data = ks_2018_modified, hue = "category")

# p.set(ylim = (0, 1000))

# p.set(xlim = (0, 100))

# plt.show()

# #カテゴリーごとの期間と達成率

# #下にオレンジが多くて、上に緑が多い？
"""

アルゴリズムを利用する

    回帰の場合は線形回帰、分類の場合はロジスティック回帰

    質的変数が扱えないアルゴリズムを使う場合は、ダミー変数に置き換える

"""

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, f1_score

from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier



X = pd.get_dummies(ks_2018_modified[["category", "launched_d", "usd_goal_real", "currency_country", "period"]], columns = ["category", "currency_country"])



X.head()
X.describe()
#目標達成率（回帰）

Y1 = ks_2018_modified["percentage"]

Y1 = pd.DataFrame(Y1, columns = ["percentage"])

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size = 0.1, random_state = 0)





#成功するかどうか（分類）

Y2 = [1 if i == "successful" else 0 for i in ks_2018["state"]]



X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size = 0.1, random_state = 0)



Y1_train.head(10)
from sklearn.preprocessing import StandardScaler



#訓練データの加工



#目標達成率（回帰）

#percentageの標準化

stdsc_percentage = StandardScaler()

Y1_train["percentage"] = stdsc_percentage.fit_transform(Y1_train[["percentage"]] .values)

#periodの標準化

stdsc_period_1 = StandardScaler()

X1_train["period"] = stdsc_period_1.fit_transform(X1_train[["period"]] .values)

#goalの標準化

stdsc_goal_1 = StandardScaler()

X1_train["usd_goal_real"] = stdsc_goal_1.fit_transform(X1_train[["usd_goal_real"]] .values)

#launched_dの標準化

stdsc_launched_1 = StandardScaler()

X1_train["launched_d"] = stdsc_launched_1.fit_transform(X1_train[["launched_d"]] .values)

#launchedとperiodの白色化

cov = np.cov(X1_train[["launched_d", "period"]], rowvar=0) # 分散・共分散を求める

_, S1 = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

launched_period = np.dot(S1.T, X1_train[["launched_d", "period"]].T).T #データを無相関化

stdsc_ld_1 = StandardScaler()

launched_period  = stdsc_ld_1.fit_transform(launched_period) # 無相関化したデータに対して、さらに標準化



#成功するかどうか（分類）

#periodの標準化

stdsc_period_2 = StandardScaler()

X2_train["period"] = stdsc_period_2.fit_transform(X2_train[["period"]] .values)

#goalの標準化

stdsc_goal_2 = StandardScaler()

X2_train["usd_goal_real"] = stdsc_goal_2.fit_transform(X2_train[["usd_goal_real"]] .values)

#launched_dの標準化

stdsc_launched_2 = StandardScaler()

X2_train["launched_d"] = stdsc_launched_2.fit_transform(X2_train[["launched_d"]] .values)

#launchedとperiodの白色化

cov = np.cov(X2_train[["launched_d", "period"]], rowvar=0) # 分散・共分散を求める

_, S2 = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

launched_period = np.dot(S2.T, X2_train[["launched_d", "period"]].T).T #データを無相関化

stdsc_ld_2 = StandardScaler()

launched_period  = stdsc_ld_2.fit_transform(launched_period) # 無相関化したデータに対して、さらに標準化
#データに相関関係がないことを確認

X1_train.corr().style.background_gradient().format('{:.2f}')
#目標達成率（回帰）

lr1 = LinearRegression(fit_intercept = True)

lr1.fit(X1_train, Y1_train)





#成功するかどうか（分類）

lr2 = LogisticRegression(random_state = 0).fit(X2_train, Y2_train) 

lr3 = Lasso(random_state = 0).fit(X2_train, Y2_train) 

lr4 = Ridge(random_state = 0).fit(X2_train, Y2_train) 

parameters = {'alpha':[0.7, 1], 'l1_ratio':[0.3, 0.7]} 

model = ElasticNet(random_state = 0)

clf = GridSearchCV(model, parameters, cv = 3)

lr5 = clf.fit(X2_train, Y2_train)

# lr5 = ElasticNet(random_state = 0).fit(X2_train, Y2_train)

#SVM

# svm_clf = SVC()

# svm_clf.fit(X2_train, Y2_train)

n_estimators = 200

svc_clf = BaggingClassifier(SVC(kernel = 'linear', probability = True, class_weight = 'balanced', random_state = 0), max_samples = 1.0 / n_estimators, n_estimators = n_estimators)

svc_clf.fit(X2_train, Y2_train)

#Random Forest

n_estimators = 100

rf_clf = RandomForestClassifier(min_samples_leaf = 20, n_estimators = n_estimators, random_state = 0)

rf_clf.fit(X2_train, Y2_train)

#AdaBoost

ab_clf = AdaBoostClassifier(random_state = 0, n_estimators = n_estimators)

ab_clf.fit(X2_train, Y2_train)
#テストデータの加工



#目標達成率（回帰）

#percentageの標準化

Y1_test["percentage"] = stdsc_percentage.transform(Y1_test[["percentage"]] .values)

#periodの標準化

X1_test["period"] = stdsc_period_1.transform(X1_test[["period"]] .values)

#goalの標準化

X1_test["usd_goal_real"] = stdsc_goal_1.transform(X1_test[["usd_goal_real"]] .values)

#launched_dの標準化

X1_train["launched_d"] = stdsc_launched_1.transform(X1_train[["launched_d"]] .values)

#launchedとdeadlineの白色化

launched_period = np.dot(S1.T, X1_test[["launched_d", "period"]].T).T #データを無相関化

launched_period  = stdsc_ld_1.transform(launched_period) # 無相関化したデータに対して、さらに標準化



#成功するかどうか（分類）

#periodの標準化

X2_test["period"] = stdsc_period_2.transform(X2_test[["period"]] .values)

#goalの標準化

X2_test["usd_goal_real"] = stdsc_goal_2.transform(X2_test[["usd_goal_real"]] .values)

#launched_dの標準化

X2_train["launched_d"] = stdsc_launched_2.transform(X2_train[["launched_d"]] .values)

#launchedとdeadlineの白色化

launched_period = np.dot(S2.T, X2_test[["launched_d", "period"]].T).T #データを無相関化

launched_period  = stdsc_ld_2.transform(launched_period) # 無相関化したデータに対して、さらに標準化
"""

予測精度または識別精度を確認する

    回帰問題の場合は、MSE、RMSE、MAEを求める

    分類問題の場合は、混同行列を作成し、Accuracy、Recall、Precisionを求める

"""

#目標達成率（回帰）

Y1_estimated = lr1.predict(X1_test)

mse = mean_squared_error(Y1_test, Y1_estimated)

rmse = np.sqrt(mse)

mae = mean_absolute_error(Y1_test, Y1_estimated)

print("MSE: {}, RMSE: {}, MAE: {}".format(mse, rmse, mae))



#成功するかどうか（分類）-- Logistic Regression + Regularization

accuracy2 = lr2.score(X2_test, Y2_test)



threshold = 0.5

accuracy3 = accuracy_score([1 if i > 0.5 else 0 for i in lr3.predict(X2_test)], Y2_test)

accuracy4 = accuracy_score([1 if i > 0.5 else 0 for i in lr4.predict(X2_test)], Y2_test)

best_clf = lr5.best_estimator_

accuracy5 = accuracy_score([1 if i > 0.5 else 0 for i in best_clf.predict(X2_test)], Y2_test)

print("Accuracy(Logistic Regression): ", accuracy2)

print("Accuracy(L1): ",                  accuracy3)

print("Accuracy(L2): ",                  accuracy4)

print("Accuracy(ElasticNet): ",          accuracy5)

cv_result = pd.DataFrame(lr5.cv_results_)

cv_result
#print(svc_clf.predict(X2_test[:10]))
#成功するかどうか（分類）-- SVM, Random Forest, Bagging

accuracy6 = accuracy_score(svc_clf.predict(X2_test), Y2_test)

print("Accuracy(Bagging): ",             accuracy6)

accuracy7 = accuracy_score(rf_clf.predict(X2_test), Y2_test)

print("Accuracy(RandomForest): ",        accuracy7)

accuracy8 = accuracy_score(ab_clf.predict(X2_test), Y2_test)

print("Accuracy(AdaBoost): ",            accuracy8)
Y2_estimated = lr2.predict(X2_test)

c_matrix = pd.DataFrame(confusion_matrix(Y2_test, Y2_estimated), 

                        index=['正解 = 成功', '正解 = 失敗'], 

                        columns=['予測 = 成功', '予測 = 失敗'])

#Recall（実際に正しいもののうち、正であると予測された割合）

recall = c_matrix["予測 = 成功"][0] / (c_matrix["予測 = 成功"][0] + c_matrix["予測 = 失敗"][0])

#Precision（正と予測したもののうち、どれくらい正しかったか）

precision = c_matrix["予測 = 成功"][0] / (c_matrix["予測 = 成功"][0] + c_matrix["予測 = 成功"][1])

#F1

f1 = f1_score(Y2_test, Y2_estimated)

print("Logistic Regression\n Recall: {}, Precioin: {}, F1: {}".format(recall, precision, f1))

c_matrix
Y2_estimated = ab_clf.predict(X2_test)

c_matrix = pd.DataFrame(confusion_matrix(Y2_test, Y2_estimated), 

                        index=['正解 = 成功', '正解 = 失敗'], 

                        columns=['予測 = 成功', '予測 = 失敗'])

#Recall（実際に正しいもののうち、正であると予測された割合）

recall = c_matrix["予測 = 成功"][0] / (c_matrix["予測 = 成功"][0] + c_matrix["予測 = 失敗"][0])

#Precision（正と予測したもののうち、どれくらい正しかったか）

precision = c_matrix["予測 = 成功"][0] / (c_matrix["予測 = 成功"][0] + c_matrix["予測 = 成功"][1])

#F1

f1 = f1_score(Y2_test, Y2_estimated)

print("AdaBoost\n Recall: {}, Precioin: {}, F1: {}".format(recall, precision, f1))

c_matrix