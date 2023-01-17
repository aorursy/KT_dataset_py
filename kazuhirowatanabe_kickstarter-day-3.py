import calendar

import datetime

import math

import re



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import pywt

import scipy.stats

from sklearn import ensemble, linear_model, metrics, model_selection
# Kaggle用

df = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")



# ローカル用

# df = pd.read_csv("kickstarter-projects/ks-projects-201801.csv")



# 分析に使用しないカラムを落とす

# ID

# 付与する規則が不明なため．

df = df.drop("ID", axis=1)



# backers

# 事後情報である．

df = df.drop("backers", axis=1)



# goal, pledged, usd_pledged

# 評価は通貨単位を統一したusd_pledged_realとusd_goal_realに対して行う．

df = df.drop("goal", axis=1)

df = df.drop("pledged", axis=1)

df = df.drop("usd pledged", axis=1)



# 評価指標achievementを設定

pledge = df["usd_pledged_real"].values

goal = df["usd_goal_real"].values

achiev = pledge / goal



df["achievement"] = achiev

df = df.drop("usd_pledged_real", axis=1)



# クリーニング

# names

# nanを落とす

df = df.dropna(subset=["name"])



# launched

# "1970-01-01 01:00:00"は欠損値扱いで削除する

df = df[df["launched"] != "1970-01-01 01:00:00"]



# state

# failedとsuccessfulだけ残す

df = df[(df["state"] == "failed") | (df["state"] == "successful")]

# 事後情報であるので説明変数にはしない．

df = df.drop("state", axis=1)



# country

# N,0"は削除

df = df[df["country"] != "N,0\""]



df
names = list(df["name"].values)

len_name = np.array([float(len(s)) for s in names])

len_name /= 100.0





df["len_name"] = len_name

df = df.drop("name", axis=1)
df = pd.get_dummies(df, columns=["category", "main_category", "country", "currency"])

df
# Warning: このセルは実行におよそ10分かかります．



def active_day(begin, end):

    date_vec = [0 for i in range(365)]

    

    delta = datetime.timedelta(days=-1)

    while(begin + delta < end):

        delta += datetime.timedelta(days=1)

        

        # 2月29日は無視する

        if (begin + delta).month == 2 and (begin + delta).day == 29:

            continue

        

        year = (begin + delta).year

        JanuaryFirst = datetime.datetime(year, 1, 1)

        index = ((begin + delta) - JanuaryFirst).days

        

        # うるう年の3月以降は2月29日を詰める

        if calendar.isleap(year) and (begin + delta).month >= 3:

            index -= 1

        

        date_vec[index] = 1

    return date_vec



deadlines = pd.to_datetime(df["deadline"])

launches = pd.to_datetime(df["launched"])

active_days = []

for d, l in zip(deadlines, launches):

    active_days.append(active_day(l, d))
# 日付のウェーブレット変換

wt_days = []

for ad in active_days:

    wt_days.append(

        np.concatenate(pywt.wavedec(ad, "db1", mode="periodic", axis=-1))

    )



fig = plt.figure(figsize=(15, 3))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

ax1.plot(range(len(active_days[0])), active_days[0])

ax2.plot(range(len(wt_days[0])), wt_days[0])

plt.show()
# 日付を直接入力する方式とウェーブレット変換をする方式を試したが，最終的にウェーブレット変換版の方が成績が良かった．

# df_day = pd.DataFrame(active_days, columns=["day_{0:03d}".format(i + 1) for i in range(len(active_days[0]))])

df_wtday = pd.DataFrame(wt_days, columns=["wt_{0:03d}".format(i) for i in range(len(wt_days[0]))])
# One-hot以外のデータの統計情報を確認

df.loc[:,["usd_goal_real", "len_name", "achievement"]].describe()
# usd_goal_realとachievementの分布を改善するため，Box-Cox変換を試す

vec_usd_goal_real = df["usd_goal_real"].values

vec_achievement = df["achievement"].values



fig = plt.figure(figsize=(15, 3))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

ax1.hist(vec_usd_goal_real, bins=32)

ax2.hist(vec_achievement, bins=32)

plt.show()



eps = 1.0e-3    # x=0におけるBox-Cox変換のエラーを回避するためのオフセット

fig = plt.figure(figsize=(15, 3))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

ax1.hist(scipy.stats.boxcox(vec_usd_goal_real + eps, lmbda=0), bins=32)

ax2.hist(scipy.stats.boxcox(vec_achievement + eps, lmbda=0), bins=32)

plt.show()



df["BC_usd_goal_real"] = scipy.stats.boxcox(vec_usd_goal_real + eps, lmbda=0)

df["BC_achievement"] = scipy.stats.boxcox(vec_achievement + eps, lmbda=0)



# メモ: 逆変換はscipy.special.inv_boxcox
# 学習データの作成

df_train = df.reset_index()

df_train = pd.concat([df_train, df_wtday], axis=1)



df_X = df_train

df_y = df["BC_achievement"]
# カラムの一覧を確認

for c in df_X.columns:

    print(c)
# 不要なカラムを落とす

df_X = df_X.drop("index", axis=1)

df_X = df_X.drop("deadline", axis=1)

df_X = df_X.drop("launched", axis=1)

df_X = df_X.drop("usd_goal_real", axis=1)

df_X = df_X.drop("achievement", axis=1)

df_X
X_train, X_test, y_train, y_test = model_selection.train_test_split(df_X.values, df_y.values, test_size=0.20, random_state=20191126)
# パラメータのグリッドサーチ

n_samples = 8192    # 全データに対してグリッドサーチをすると時間がかかりすぎるため

                    # データ数を制限する．

param_n_estimators = [int(8 * 2**i) for i in range(6)]

param_min_samples_split = [int(2 * 2**i) for i in range(4)]

for n_estimators in param_n_estimators:

    for min_samples_split in param_min_samples_split:

        estimator = ensemble.RandomForestRegressor(

            n_estimators=n_estimators,

            min_samples_split = min_samples_split,

            criterion="mse",

            bootstrap=True,

            n_jobs=-1

        )

        kf = model_selection.KFold(n_splits=8, shuffle=True, random_state=20191128)

        scores = model_selection.cross_validate(

            estimator,

            X_train[:n_samples],

            y_train[:n_samples],

            cv=kf,

            scoring="neg_mean_squared_error"

        )

        print("Test sores: n-est. = {0:d}, min. split={1:d}".format(n_estimators, min_samples_split))

        print("1:{0:10.3e}, 2:{1:10.3e}, 2:{2:10.3e}, 2:{3:10.3e}".format(

            scores["test_score"][0],

            scores["test_score"][1],

            scores["test_score"][2],

            scores["test_score"][3],

        ))

        print("5:{0:10.3e}, 6:{1:10.3e}, 7:{2:10.3e}, 8:{3:10.3e}".format(

            scores["test_score"][4],

            scores["test_score"][5],

            scores["test_score"][6],

            scores["test_score"][7],

        ))

        print("Mean KFold score: {0:10.3e}".format(scores["test_score"].mean()))

        print()
# Testデータに対して答え合わせ

estimator = ensemble.RandomForestRegressor(

    n_estimators=256,

    min_samples_split = 2,

    criterion="mse",

    bootstrap=True,

    n_jobs=-1

)



estimator.fit(X_train, y_train)

y_fit  = estimator.predict(X_train)

y_pred = estimator.predict(X_test)



# Box-coxをもとに戻す

y_train_real = scipy.special.inv_boxcox(y_train, 0) - eps   # 教師データ

y_fit_real   = scipy.special.inv_boxcox(y_fit  , 0) - eps   # 教師データに対する予測

y_test_real  = scipy.special.inv_boxcox(y_test , 0) - eps   # 検証データ

y_pred_real  = scipy.special.inv_boxcox(y_pred , 0) - eps   # 検証データに対する予測



RMSE_train = math.sqrt(metrics.mean_squared_error(y_train_real, y_fit_real))

MAE_train  = metrics.mean_absolute_error(y_train_real, y_fit_real)

RMSE_test  = math.sqrt(metrics.mean_squared_error(y_test_real, y_pred_real))

MAE_test   = metrics.mean_absolute_error(y_test_real, y_pred_real)



print("[Train] Root mean squared error: {0:.0f}".format(RMSE_train))

print("[Train] Mean absolute error    : {0:.0f}".format(MAE_train))

print("[Test]  Root mean squared error: {0:.0f}".format(RMSE_test))

print("[Test]  Mean absolute error    : {0:.0f}".format(MAE_test))
