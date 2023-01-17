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
import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression





# 多元线性回归+特征工程，参考https://blog.csdn.net/weixin_39739342/article/details/93379653





# 从原始数据中，构造数据列，参考https://www.kaggle.com/kernels/scriptcontent/6613593/download

def rank_by_team(df):

    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc', "matchType"]

    features = [col for col in df.columns if col not in cols_to_drop]

    agg = df.groupby(['matchId', 'groupId'])[features].mean()

    agg = agg.groupby('matchId')[features].rank(pct=True)

    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])





df = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')



# 删除缺失值

df = df.dropna()



# 增加数据列，提高精度

df = rank_by_team(df)



# 非数字的列

cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc', "matchType"]

# 定义特征

features = [col for col in df.columns if col not in cols_to_drop]



# 对全部标签进行回归

# X = df.loc[:, ("assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", "killPlace",

#                "killPoints", "kills", "killStreaks", "longestKill", "matchDuration", "maxPlace",

#                "numGroups", "rankPoints", "revives", "rideDistance", "roadKills", "swimDistance", "teamKills",

#                "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints")]

X = df.loc[:, features]

Y = df.loc[:, "winPlacePerc"]



# 这里是不是可以将全部的数据划分为训练数据？要留一部分testdata吗

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

X_train = X

# Y_train = Y.values.reshape(-1,1)

Y_train = Y



# # 归一化

# min_max_scaler = preprocessing.MinMaxScaler()

# X_train = min_max_scaler.fit_transform(X_train)





model = LinearRegression()

model.fit(X_train, Y_train)

# print(model.coef_)

# print(model.intercept_)



test_data = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")



# 增加数据列，提高精度

test_data = rank_by_team(test_data)



# 对全部标签进行回归

# X = test_data.loc[:, ("assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", "killPlace",

#                       "killPoints", "kills", "killStreaks", "longestKill", "matchDuration", "maxPlace",

#                       "numGroups", "rankPoints", "revives", "rideDistance", "roadKills", "swimDistance", "teamKills",

#                       "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints")]



X = test_data.loc[:, features]

# X = min_max_scaler.fit_transform(X)



pred = model.predict(X)



id = pd.read_csv("../input/pubg-finish-placement-prediction/sample_submission_V2.csv")

id = id.loc[:, "Id"]



# pred = pred.flatten()

df = pd.DataFrame({"Id": id, "winPlacePerc": pred})

df.to_csv("submission.csv", index=False, sep=',')

import numpy as np

import pandas as pd



# 简单的数据后处理，参考https://www.kaggle.com/ceshine/a-simple-post-processing-trick-lb-0237-0204/code



df_sub = pd.read_csv("submission.csv")

df_test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")



# 将结果与训练数据拼接

df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")



# 排序，分配调整后的比率

df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()

df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()

df_sub_group = df_sub_group.merge(

    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(),

    on="matchId", how="left")

df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)



df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")

df_sub["winPlacePerc"] = df_sub["adjusted_perc"]



# 对极端情况（边界情况）进行处理，防止有越界的数据

df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0

df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1



# 与maxPlace对齐

subset = df_sub.loc[df_sub.maxPlace > 1]

gap = 1.0 / (subset.maxPlace.values - 1)

new_perc = np.around(subset.winPlacePerc.values / gap) * gap

df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc



# 对边界情况进行处理

df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0

assert df_sub["winPlacePerc"].isnull().sum() == 0



df_sub[["Id", "winPlacePerc"]].to_csv("submission.csv", index=False)
