import datetime as dt

import math



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn import linear_model, metrics, model_selection
df = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")



# 分析に使用しないカラムを落とす

# ID

# 付与する規則が不明なため．

df = df.drop("ID", axis=1)



# goal, pledged, usd_pledged

# 通貨単位を統一したusd_pledged_realとusd_goal_realで評価する．

df = df.drop("goal", axis=1)

df = df.drop("pledged", axis=1)

df = df.drop("usd pledged", axis=1)



pledge = df["usd_pledged_real"].values

goal = df["usd_goal_real"].values

achiev = pledge / goal



# 評価指標を設定

df["achievement"] = achiev

df = df.drop("usd_pledged_real", axis=1)



# クリーニング

# launched

# "1970-01-01 01:00:00"は欠損値扱いで削除する

df = df[df["launched"] != "1970-01-01 01:00:00"]



# state

# failedとsuccessfulだけ残す

df = df[(df["state"] == "failed") | (df["state"] == "successful")]



# country

# N,0"は削除

df = df[df["country"] != "N,0\""]



df
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



df_dummies
# 説明変数

explanatories = [

    "main_category_Art",    # 0

    "main_category_Comics",    # 1

    "main_category_Crafts",    # 2

    "main_category_Dance",    # 3

    "main_category_Design",    # 4

    "main_category_Fashion",    # 5

    "main_category_Film & Video",    # 6

    "main_category_Food",    # 7

    "main_category_Games",    # 8

    "main_category_Journalism",    # 9

    "main_category_Music",    # 10

    "main_category_Photography",    # 11

    "main_category_Publishing",    # 12

    "main_category_Technology",    # 13

    "main_category_Theater",    # 14

    "usd_goal_real",    # 15

    "deadline_unix",    # 16

    "launched_unix",    # 17

    "country_AT",    # 18

    "country_AU",    # 19

    "country_BE",    # 20

    "country_CA",    # 21

    "country_CH",    # 22

    "country_DE",    # 23

    "country_DK",    # 24

    "country_ES",    # 25

    "country_FR",    # 26

    "country_GB",    # 27

    "country_HK",    # 28

    "country_IE",    # 29

    "country_IT",    # 30

    "country_JP",    # 31

    "country_LU",    # 32

    "country_MX",    # 33

    "country_NL",    # 34

    "country_NO",    # 35

    "country_NZ",    # 36

    "country_SE",    # 37

    "country_SG",    # 38

    "country_US",    # 39

]

X = df_dummies.loc[:, explanatories].values

y = df_dummies["achievement"].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=20191117)
eps = 1.0e-2

fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132, sharey=ax1)

ax3 = fig.add_subplot(133, sharey=ax1)

ax1.hist(np.log10(y + eps), bins=32)

ax2.hist(np.log10(y_train + eps), bins=32)

ax3.hist(np.log10(y_test + eps), bins=32)
# usd_goal_real

usd_goal_real_trains = []

for i in range(len(X_train)):

    usd_goal_real_trains.append(X_train[i][15])

arr = np.array(usd_goal_real_trains)

usd_goal_real_mean = arr.mean()

usd_goal_real_std = arr.std()



for i in range(len(X_train)):

    X_train[i][15] = (X_train[i][15] - usd_goal_real_mean)/usd_goal_real_std

for i in range(len(X_test)):

    X_test[i][15] = (X_test[i][15] - usd_goal_real_mean)/usd_goal_real_std



# deadline

timestamp_min = dt.datetime(2009, 1, 1).timestamp()

timestamp_max = dt.datetime(2019, 1, 1).timestamp()

for i in range(len(X_train)):

    X_train[i][16] = (X_train[i][16] - timestamp_min)/(timestamp_max - timestamp_min)

for i in range(len(X_test)):

    X_test[i][16] = (X_test[i][16] - timestamp_min)/(timestamp_max - timestamp_min)



# launched

timestamp_min = dt.datetime(2009, 1, 1).timestamp()

timestamp_max = dt.datetime(2019, 1, 1).timestamp()

for i in range(len(X_train)):

    X_train[i][17] = (X_train[i][17] - timestamp_min)/(timestamp_max - timestamp_min)

for i in range(len(X_test)):

    X_test[i][17] = (X_test[i][17] - timestamp_min)/(timestamp_max - timestamp_min)



# achievement

arr = np.array(y_train)

achievement_mean = arr.mean()

achievement_std = arr.std()



for i in range(len(y_train)):

    y_train[i] = (y_train[i] - achievement_mean)/achievement_std

for i in range(len(X_test)):

    y_test[i]= (y_test[i] - achievement_mean)/achievement_std



# achievementの標準化を解除する関数を書いておく

def unstandalize_achievement(achiev):

    unstandalized = []

    for v in achiev:

        unstandalized.append(v * achievement_std + achievement_mean)

    return unstandalized
for a in [10**(i/4 + 2) for i in range(16 + 1)]:

    RidgeLRPred = linear_model.Ridge(alpha=a)

    RidgeLRPred.fit(X_train, y_train)

    y_train_pred = RidgeLRPred.predict(X_train)

    y_test_pred = RidgeLRPred.predict(X_test)

    

    MSE_train = metrics.mean_squared_error(

        unstandalize_achievement(y_train), unstandalize_achievement(y_train_pred))

    MSE_test = metrics.mean_squared_error(

        unstandalize_achievement(y_test), unstandalize_achievement(y_test_pred))

    print("Alpha={0:10.3e}, MSE_train={1:10.3e}, MSE_test={2:10.3e}".format(a, MSE_train, MSE_test))