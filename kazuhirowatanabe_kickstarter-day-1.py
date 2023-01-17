import datetime as dt

import math



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn import linear_model, metrics
df = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

df.head()
# ID

# 付与する規則が不明なため．

df = df.drop("ID", axis=1)



# goal, pledged, usd_pledged

# 通貨単位を統一したusd_pledged_realとusd_goal_realで評価する．

df = df.drop("goal", axis=1)

df = df.drop("pledged", axis=1)

df = df.drop("usd pledged", axis=1)



df.head()
pledge = df["usd_pledged_real"].values

goal = df["usd_goal_real"].values

achiev = pledge / goal



df["achievement"] = achiev

df = df.drop("usd_pledged_real", axis=1)

df.head()
def show_unique_items(s):

    items = list(s.unique())

    items.sort()

    for index, item in enumerate(items):

        print("{0:6d}: {1}".format(index, item))

    return



# category

show_unique_items(df["category"])
# main_category

show_unique_items(df["main_category"])
# currency

show_unique_items(df["currency"])
# deadline

show_unique_items(df["deadline"])
# launched

show_unique_items(df["launched"])
# "1970-01-01 01:00:00"は欠損値扱いで削除する

df = df[df["launched"] != "1970-01-01 01:00:00"]
# state

show_unique_items(df["state"])
# failedとsuccessfulだけ残す

df = df[(df["state"] == "failed") | (df["state"] == "successful")]
# backers

show_unique_items(df["backers"])
# country

show_unique_items(df["country"])
# N,0"は削除

df = df[df["country"] != "N,0\""]
# usd_goal_real

show_unique_items(df["usd_goal_real"])
# achievement

show_unique_items(df["achievement"])
# ヒストグラム

fig = plt.figure()

ax = fig.add_subplot(111)

ax.hist(df["achievement"].values, bins=32)

ax.set_xlabel("Achievement")

plt.show()
eps = 1.0e-2    # log10(0)を避けるためのオフセット．

                # 達成率1%以下は区別しない．

fig = plt.figure()

ax = fig.add_subplot(111)

ax.hist(np.log10(df["achievement"].values + eps), bins=32)

ax.set_xlabel("log10(Achievement)")

plt.show()
def deadline_str_to_int(array):

    utv = []

    for s in array:

        datetime_obj = dt.datetime.strptime(s, "%Y-%m-%d")

        utv.append(datetime_obj.timestamp())

    return utv



def launched_str_to_int(array):

    utv = []

    for s in array:

        datetime_obj = dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

        utv.append(datetime_obj.timestamp())

    return utv



df_dummies = pd.get_dummies(df, columns=["main_category", "country"])

df_dummies["deadline_unix"] = deadline_str_to_int(df["deadline"].values)

df_dummies["launched_unix"] = launched_str_to_int(df["launched"].values)

df_dummies.columns
# 説明変数

explanatories = [

    "main_category_Art",

    "main_category_Comics",

    "main_category_Crafts",

    "main_category_Dance",

    "main_category_Design",

    "main_category_Fashion",

    "main_category_Film & Video",

    "main_category_Food",

    "main_category_Games",

    "main_category_Journalism",

    "main_category_Music",

    "main_category_Photography",

    "main_category_Publishing",

    "main_category_Technology",

    "main_category_Theater",

    "usd_goal_real",

    "deadline_unix",

    "launched_unix",

    "country_AT",

    "country_AU",

    "country_BE",

    "country_CA",

    "country_CH",

    "country_DE",

    "country_DK",

    "country_ES",

    "country_FR",

    "country_GB",

    "country_HK",

    "country_IE",

    "country_IT",

    "country_JP",

    "country_LU",

    "country_MX",

    "country_NL",

    "country_NO",

    "country_NZ",

    "country_SE",

    "country_SG",

    "country_US",

] 

X = df_dummies.loc[:, explanatories].values

y = df_dummies["achievement"].values
# 線形回帰

LRPred = linear_model.LinearRegression()

LRPred.fit(X, y)
for exp, coef in zip(explanatories, LRPred.coef_):

    print("{0:26s}: {1:10.3e}".format(exp, coef))

print("{0:26s}: {1:10.3e}".format("intercept", LRPred.intercept_))
y_pred = LRPred.predict(X)

MAE = metrics.mean_absolute_error(y, y_pred)

MSE = metrics.mean_squared_error(y, y_pred)

print("Mean absolute error:     {0:.2f}".format(MAE))

print("Root mean squared error: {0:.2f}".format(math.sqrt(MSE)))