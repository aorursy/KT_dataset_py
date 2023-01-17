import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

%matplotlib inline
train = pd.read_csv("../input/train_set.csv", date_parser="SALEDATE")

test = pd.read_csv("../input/test_set.csv", date_parser="SALEDATE")

print("train shape:", train.shape)

print("test shape:", test.shape)
#目的関数の対数を準備

train["log_PRICE"] = np.log(train["PRICE"])

#この時点で目的変数が外れ値のものはのぞいておく（3件）

train = train[train["log_PRICE"] != 0]

#モデルに使用しない予定の変数はここで落としておく

train = train.drop(["FULLADDRESS", "CITY", "STATE", "ZIPCODE", "NATIONALGRID", "ASSESSMENT_NBHD","GIS_LAST_MOD_DTTM", 

                                "ASSESSMENT_SUBNBHD", "CENSUS_TRACT", "CENSUS_BLOCK", "X", "Y", "PRICE"], axis=1)





train_C = train[train["SOURCE"] =="Condominium"]

train_R = train[train["SOURCE"] =="Residential"]

print("shape of C:", train_C.shape)

print("shape of R:", train_R.shape)
def miss_kakunin(data):

    missing = data.isnull().sum()

    missing = missing[missing > 0]

    missing = missing.sort_values(ascending=False)

    missing.plot.bar()

    return missing
miss_kakunin(train_C)
#20051個欠損がある変数は使用しない

train_C["AYB"] = np.where(train_C["AYB"].isnull(), train_C["AYB"].mode(), train_C["AYB"])

train_C["YR_RMDL"] = np.where(train_C["YR_RMDL"].isnull(), train_C["AYB"], train_C["YR_RMDL"]) #リモデルしてない物件は建築年で埋める)



#使用しない変数を落とす(Residentialのみの変数)

train_C = train_C.drop(["ROOF", "STORIES", "GBA", "STYLE", "STRUCT", "GRADE", "CNDTN", 

                                          "EXTWALL", "INTWALL", "KITCHENS","NUM_UNITS"], axis=1)



#使用しない変数を落とす（意味不明なもの）

train_C = train_C.drop(["Id", "SALE_NUM", "BLDG_NUM", "USECODE", "SOURCE", "CMPLX_NUM"], axis=1)
"""

命名規則

# 変数を丸めたり少し加工しただけのものは変数名+数字

# 変数の組み合わせで新変数を作成しているものは接頭NEW_

"""

#数値変数

train_C["HF_BATHRM2"] =  np.where(train_C["HF_BATHRM"] >= 3, 3, train_C["HF_BATHRM"]) 

train_C["NEW_total_BATHRM"] =  train_C["BATHRM"] + train_C["HF_BATHRM"]

train_C["ROOMS2"] = np.where(train_C["ROOMS"] >= 10, 10, train_C["ROOMS"])

train_C["NEW_BED/ROOMS"] = np.where(train_C["ROOMS2"]!=0, train_C["BEDRM"] / train_C["ROOMS2"], 0)

train_C["AYB2"] = np.abs(1950 - train_C["AYB"])

train_C["YR_RMDL2"] = np.where(train_C["YR_RMDL"] == 20, train_C["AYB2"], train_C["YR_RMDL"])

train_C["NEW_AYB_to_RMDL"] = train_C["YR_RMDL2"] - train_C["AYB2"]

train_C["NEW_AYB_to_EYB"] = train_C["EYB"] - train_C["AYB2"]

train_C["NEW_EYB_to_YR_RMDL"] = np.abs(np.abs(train_C["YR_RMDL2"] - train_C["EYB"]) - 60)

train_C["FIREPLACES2"] = np.where(train_C["FIREPLACES"] > 5, 5, train_C["FIREPLACES"])

train_C["SQUARE"] = train_C["SQUARE"].astype("int") #CのSQUAREはそののママで数値に変換できる

train_C["SALEDATE_Y"] = pd.to_datetime(train_C["SALEDATE"]).dt.year

#数値変数(追加)

train_C["NEW_LANDAREA/LIVING_GBA"] = np.where(train_C["LIVING_GBA"]==0, 0, train_C["LANDAREA"] / train_C["LIVING_GBA"])

train_C["NEW_RMDL_to_SALE"] = train_C["SALEDATE_Y"] - train_C["YR_RMDL2"]

train_C["NEW_SQUARE/LIVING_GBA"] = np.where(train_C["LIVING_GBA"]==0, 0, train_C["SQUARE"] / train_C["LIVING_GBA"])

train_C["NEW_BATH/ROOMS"] = np.where(train_C["ROOMS2"]==0, 0, train_C["NEW_total_BATHRM"] / train_C["ROOMS2"])



#カテゴリ変数

train_C["AC2"] = np.where(train_C["AC"]=="0", "N", train_C["AC"])

train_C["dum_AC_Y"] = np.where(train_C["AC2"]=="Y", 1, 0)

train_C["dum_QUALIFIED_Q"] = np.where(train_C["QUALIFIED"]=="Q", 1, 0)

train_C["dum_WARD_1"] = np.where(train_C["WARD"]=="Ward 1", 1, 0)

train_C["dum_WARD_2"] = np.where(train_C["WARD"]=="Ward 2", 1, 0)

train_C["dum_WARD_3"] = np.where(train_C["WARD"]=="Ward 3", 1, 0)

train_C["dum_WARD_4"] = np.where(train_C["WARD"]=="Ward 4", 1, 0)

train_C["dum_WARD_5"] = np.where(train_C["WARD"]=="Ward 5", 1, 0)

train_C["dum_WARD_6"] = np.where(train_C["WARD"]=="Ward 6", 1, 0)

train_C["dum_WARD_7"] = np.where(train_C["WARD"]=="Ward 7", 1, 0)

train_C["dum_QUADRANT_NW"] = np.where(train_C["QUADRANT"]=="NW", 1, 0)

train_C["dum_QUADRANT_SE"] = np.where(train_C["QUADRANT"]=="SE", 1, 0)

train_C["dum_QUADRANT_NE"] = np.where(train_C["QUADRANT"]=="NE", 1, 0)



#変換元の変数は落とす(HEATは後で考える)

train_C = train_C.drop(["HF_BATHRM", "ROOMS", "AYB", "YR_RMDL", "FIREPLACES", "SALEDATE", "AC", "AC2", "QUALIFIED", "WARD", "QUADRANT", "HEAT"], axis=1)
#変数同士の相関を確認

plt.figure(figsize=(18,18))

sns.heatmap(train_C.corr('pearson'),

            annot=True,

            fmt='.2f',

            vmin=-1,

            vmax=1,

            cmap="seismic")
#相関係数が高い変数を落とす

train_C = train_C.drop(["BATHRM", "BEDRM", "EYB"], axis=1)
#学習用にデータを整形

C_y = train_C["log_PRICE"]

C_X = train_C.drop("log_PRICE", axis=1)



#手元で精度確認用にtrain testに分割

C_tr_X, C_te_X, C_tr_y, C_te_y = train_test_split(C_X, C_y, test_size=0.3, random_state=42)

print("shape of C_tr_X:", C_tr_X.shape)

print("shape of C_tr_y:", C_tr_y.shape)

print("shape of C_te_X:", C_te_X.shape)

print("shape of C_te_y:", C_te_y.shape)
#モデルを作成(パラメータはとりあえず適当)

clf_C = XGBRegressor(max_depth=8, n_estimators=5000, seed=3, learning_rate=0.01)

clf_C
clf_C.fit(C_tr_X, C_tr_y, early_stopping_rounds=15, eval_set=[[C_te_X, C_te_y]])
#変数重要度の確認

feature_importances = pd.DataFrame(clf_C.feature_importances_,

                                   index = C_tr_X.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)

print(feature_importances.head(20))



feature_importances = feature_importances.sort_values('importance',ascending=True).tail(20)

plt.barh(feature_importances.index, feature_importances.importance)
miss_kakunin(train_R)
#SALEDATE欠損先は削除する(1件だし、重要度が高いため)

train_R = train_R[train_R["SALEDATE"].notnull()]



#29054個欠損がある変数は使用しない

train_R["AYB"] = np.where(train_R["AYB"].isnull(), train_R["AYB"].mode(), train_R["AYB"])

train_R["YR_RMDL"] = np.where(train_R["YR_RMDL"].isnull(), train_R["AYB"], train_R["YR_RMDL"]) #リモデルしてない物件は建築年で埋める)

train_R["QUADRANT"] = np.where(train_R["QUADRANT"].isnull(), train_R["QUADRANT"].mode(), train_R["QUADRANT"])

train_R["STORIES"] = np.where(train_R["STORIES"].isnull(), train_R["STORIES"].mode(), train_R["STORIES"])







#使用しない変数を落とす(Condominiamのみの変数)

train_R = train_R.drop(["LIVING_GBA", "CMPLX_NUM"], axis=1)



#使用しない変数を落とす（意味不明なもの）

train_R = train_R.drop(["Id", "SALE_NUM", "BLDG_NUM", "USECODE", "SOURCE"], axis=1)
def GRADE_to_num(x):

    if x == "Low Quality":

        y = 0

    elif x == "Fair Quality":

        y = 1

    elif x == "Average":

        y = 2

    elif x == "Above Average":

        y = 3

    elif x == "Good Quality":

        y = 4

    elif x == "Very Good":

        y = 5

    elif x == "Excellent":

        y = 6

    elif x == "Superior":

        y = 7

    elif x == "Exceptional-A":

        y = 8

    elif x == "Exceptional-B":

        y = 9

    elif x == "Exceptional-C":

        y = 10

    elif x == "Exceptional-D":

        y = 11

    elif x == "No Data": #test_Rに潜んでいるので注意

        y = -1

    return y



def CNDTN_to_num(x):

    if x == "Poor":

        y = 0

    elif x == "Fair":

        y = 1

    elif x == "Average":

        y = 2

    elif x == "Good":

        y = 3

    elif x == "Very Good":

        y = 4

    elif x == "Excellent":

        y = 5

    elif x == "Default": #test_Rに潜んでいるので注意

        y = 2

    return y
#数値変数

train_R["HF_BATHRM2"] =  np.where(train_R["HF_BATHRM"] >= 3, 3, train_R["HF_BATHRM"])

train_R["NEW_total_BATHRM"] =  train_R["BATHRM"] + train_R["HF_BATHRM"]

train_R["ROOMS2"] = np.where(train_R["ROOMS"] >= 10, 10, train_R["ROOMS"])

train_R["NEW_BED/ROOMS"] = np.where(train_R["ROOMS2"]!=0, train_R["BEDRM"] / train_R["ROOMS2"], 0)

train_R["AYB2"] = np.abs(1950 - train_R["AYB"])

train_R["YR_RMDL2"] = np.where(train_R["YR_RMDL"] == 20, train_R["AYB2"], train_R["YR_RMDL"])

train_R["STORIES2"] = np.where(train_R["STORIES"] > 2, 2, train_R["STORIES"])

train_R["NEW_LAND/GBA"] = train_R["LANDAREA"] / train_R["GBA"]

train_R["NEW_AYB_to_RMDL"] = train_R["YR_RMDL2"] - train_R["AYB2"]

train_R["NEW_AYB_to_EYB"] = train_R["EYB"] - train_R["AYB2"]

train_R["NEW_EYB_to_YR_RMDL"] = np.abs(np.abs(train_R["YR_RMDL2"] - train_R["EYB"]) - 60)

train_R["KITCHENS2"] = np.where(train_R["KITCHENS"] > 5, 5, train_R["KITCHENS"])

train_R["KITCHENS3"] = np.abs(train_R["KITCHENS2"] - 3) ** 2

train_R["FIREPLACES2"] = np.where(train_R["FIREPLACES"] > 5, 5, train_R["FIREPLACES"])

train_R["SQUARE"] = np.where(train_R["SQUARE"] == "PAR ", train_R["SQUARE"].mode(), train_R["SQUARE"])

train_R["SQUARE"] = train_R["SQUARE"].astype("int") #"PAR "を置換してから数値にする

train_R["SALEDATE_Y"] = pd.to_datetime(train_R["SALEDATE"]).dt.year

#数値変数(追加)

train_R["NEW_RMDL_to_SALE"] = train_R["SALEDATE_Y"] - train_R["YR_RMDL2"]

train_R["NEW_SQUARE/GBA"] = np.where(train_R["GBA"]==0, 0, train_R["SQUARE"] / train_R["GBA"])

train_R["NEW_BATH/ROOMS"] = np.where(train_R["ROOMS2"]==0, 0, train_R["NEW_total_BATHRM"] / train_R["ROOMS2"])





#カテゴリ変数

#(HEAT, STYLE、STRUCT,EXTWALL,ROOF,INTWALL(cadinaryが多いやつ)はあとで考える)

train_R["AC2"] = np.where(train_R["AC"]=="0", "N", train_R["AC"])

train_R["dum_AC_Y"] = np.where(train_R["AC2"]=="Y", 1, 0)

train_R["dum_QUALIFIED_Q"] = np.where(train_R["QUALIFIED"]=="Q", 1, 0)

train_R["dum_WARD_1"] = np.where(train_R["WARD"]=="Ward 1", 1, 0)

train_R["dum_WARD_2"] = np.where(train_R["WARD"]=="Ward 2", 1, 0)

train_R["dum_WARD_3"] = np.where(train_R["WARD"]=="Ward 3", 1, 0)

train_R["dum_WARD_4"] = np.where(train_R["WARD"]=="Ward 4", 1, 0)

train_R["dum_WARD_5"] = np.where(train_R["WARD"]=="Ward 5", 1, 0)

train_R["dum_WARD_6"] = np.where(train_R["WARD"]=="Ward 6", 1, 0)

train_R["dum_WARD_7"] = np.where(train_R["WARD"]=="Ward 7", 1, 0)

train_R["dum_QUADRANT_NW"] = np.where(train_R["QUADRANT"]=="NW", 1, 0)

train_R["dum_QUADRANT_SE"] = np.where(train_R["QUADRANT"]=="SE", 1, 0)

train_R["dum_QUADRANT_NE"] = np.where(train_R["QUADRANT"]=="NE", 1, 0)

train_R["GRADE"] = train_R["GRADE"].apply(GRADE_to_num)

train_R["CNDTN"] = train_R["CNDTN"].apply(CNDTN_to_num)



#変換元の変数は落とす

train_R = train_R.drop(["HF_BATHRM", "ROOMS", "AYB", "YR_RMDL", "FIREPLACES", "SALEDATE", "AC", "AC2", "QUALIFIED", "WARD", "QUADRANT", "HEAT"], axis=1)

train_R = train_R.drop(["STYLE", "STRUCT", "EXTWALL", "INTWALL", "ROOF"], axis=1)
#変数同士の相関を確認

plt.figure(figsize=(18,18))

sns.heatmap(train_R.corr('pearson'),

            annot=True,

            fmt='.2f',

            vmin=-1,

            vmax=1,

            cmap="seismic")
#相関係数が高い変数を落とす

train_R = train_R.drop(["NUM_UNITS", "KITCHENS", "KITCHENS2", "YR_RMDL2"], axis=1)
#学習用にデータを整形

R_y = train_R["log_PRICE"]

R_X = train_R.drop("log_PRICE", axis=1)



#手元で精度確認用にtrain testに分割

R_tr_X, R_te_X, R_tr_y, R_te_y = train_test_split(R_X, R_y, test_size=0.3, random_state=42)

print("shape of R_tr_X:", C_tr_X.shape)

print("shape of R_tr_y:", C_tr_y.shape)

print("shape of R_te_X:", C_te_X.shape)

print("shape of R_te_y:", C_te_y.shape)
#念のためXGBで確認(パラメータはとりあえず適当)

clf_R = XGBRegressor(max_depth=8, n_estimators=5000, seed=3, learning_rate=0.01)

clf_R
clf_R.fit(R_tr_X, R_tr_y, early_stopping_rounds=15, eval_set=[[R_te_X, R_te_y]])
#変数重要度の確認

feature_importances = pd.DataFrame(clf_R.feature_importances_,

                                   index = R_tr_X.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)

print(feature_importances.head(20))



feature_importances = feature_importances.sort_values('importance',ascending=True).tail(20)

plt.barh(feature_importances.index, feature_importances.importance)
#モデルに使用しない予定の変数はここで落としておく

test = test.drop(["FULLADDRESS", "CITY", "STATE", "ZIPCODE", "NATIONALGRID", "ASSESSMENT_NBHD","GIS_LAST_MOD_DTTM", 

                                "ASSESSMENT_SUBNBHD", "CENSUS_TRACT", "CENSUS_BLOCK", "X", "Y"], axis=1)





test_C = test[test["SOURCE"] =="Condominium"]

test_R = test[test["SOURCE"] =="Residential"]

print("shape of C:", test_C.shape)

print("shape of R:", test_R.shape)
miss_kakunin(test_C)
#統計値で欠損を埋める時はtrainの統計値を使うことに注意する（→謎のエラーが出るのでとりあえずtestで埋めておく）

test_C["AYB"] = np.where(test_C["AYB"].isnull(), test_C["AYB"].mode(), test_C["AYB"])

test_C["YR_RMDL"] = np.where(test_C["YR_RMDL"].isnull(), test_C["AYB"], test_C["YR_RMDL"]) #リモデルしてない物件は建築年で埋める)



#使用しない変数を落とす(Residentialのみの変数)

test_C = test_C.drop(["ROOF", "STORIES", "GBA", "STYLE", "STRUCT", "GRADE", "CNDTN", 

                                          "EXTWALL", "INTWALL", "KITCHENS","NUM_UNITS"], axis=1)



#使用しない変数を落とす（意味不明なもの）

test_C = test_C.drop(["Id", "SALE_NUM", "BLDG_NUM", "USECODE", "SOURCE", "CMPLX_NUM"], axis=1)
#数値変数

test_C["HF_BATHRM2"] =  np.where(test_C["HF_BATHRM"] >= 3, 3, test_C["HF_BATHRM"]) 

test_C["NEW_total_BATHRM"] =  test_C["BATHRM"] + test_C["HF_BATHRM"]

test_C["ROOMS2"] = np.where(test_C["ROOMS"] >= 10, 10, test_C["ROOMS"])

test_C["NEW_BED/ROOMS"] = np.where(test_C["ROOMS2"]!=0, test_C["BEDRM"] / test_C["ROOMS2"], 0)

test_C["AYB2"] = np.abs(1950 - test_C["AYB"])

test_C["YR_RMDL2"] = np.where(test_C["YR_RMDL"] == 20, test_C["AYB2"], test_C["YR_RMDL"])

test_C["NEW_AYB_to_RMDL"] = test_C["YR_RMDL2"] - test_C["AYB2"]

test_C["NEW_AYB_to_EYB"] = test_C["EYB"] - test_C["AYB2"]

test_C["NEW_EYB_to_YR_RMDL"] = np.abs(np.abs(test_C["YR_RMDL2"] - test_C["EYB"]) - 60)

test_C["FIREPLACES2"] = np.where(test_C["FIREPLACES"] > 5, 5, test_C["FIREPLACES"])

test_C["SQUARE"] = test_C["SQUARE"].astype("int") #CのSQUAREはそののママで数値に変換できる

test_C["SALEDATE_Y"] = pd.to_datetime(test_C["SALEDATE"]).dt.year

#追加分

test_C["NEW_LANDAREA/LIVING_GBA"] = np.where(test_C["LIVING_GBA"]==0, 0, test_C["LANDAREA"] / test_C["LIVING_GBA"])

test_C["NEW_RMDL_to_SALE"] = test_C["SALEDATE_Y"] - test_C["YR_RMDL2"]

test_C["NEW_SQUARE/LIVING_GBA"] = np.where(test_C["LIVING_GBA"]==0, 0, test_C["SQUARE"] / test_C["LIVING_GBA"])

test_C["NEW_BATH/ROOMS"] = np.where(test_C["ROOMS2"]==0, 0, test_C["NEW_total_BATHRM"] / test_C["ROOMS2"])



#カテゴリ変数

test_C["AC2"] = np.where(test_C["AC"]=="0", "N", test_C["AC"])

test_C["dum_AC_Y"] = np.where(test_C["AC2"]=="Y", 1, 0)

test_C["dum_QUALIFIED_Q"] = np.where(test_C["QUALIFIED"]=="Q", 1, 0)

test_C["dum_WARD_1"] = np.where(test_C["WARD"]=="Ward 1", 1, 0)

test_C["dum_WARD_2"] = np.where(test_C["WARD"]=="Ward 2", 1, 0)

test_C["dum_WARD_3"] = np.where(test_C["WARD"]=="Ward 3", 1, 0)

test_C["dum_WARD_4"] = np.where(test_C["WARD"]=="Ward 4", 1, 0)

test_C["dum_WARD_5"] = np.where(test_C["WARD"]=="Ward 5", 1, 0)

test_C["dum_WARD_6"] = np.where(test_C["WARD"]=="Ward 6", 1, 0)

test_C["dum_WARD_7"] = np.where(test_C["WARD"]=="Ward 7", 1, 0)

test_C["dum_QUADRANT_NW"] = np.where(test_C["QUADRANT"]=="NW", 1, 0)

test_C["dum_QUADRANT_SE"] = np.where(test_C["QUADRANT"]=="SE", 1, 0)

test_C["dum_QUADRANT_NE"] = np.where(test_C["QUADRANT"]=="NE", 1, 0)



#変換元の変数は落とす(HEATは後で考える)

test_C = test_C.drop(["HF_BATHRM", "ROOMS", "AYB", "YR_RMDL", "FIREPLACES", "SALEDATE", "AC", "AC2", "QUALIFIED", "WARD", "QUADRANT", "HEAT"], axis=1)
#相関係数が高い変数を落とす

test_C = test_C.drop(["BATHRM", "BEDRM", "EYB"], axis=1)
#モデル当てはめ

pred_C_XGB = clf_C.predict(test_C)
miss_kakunin(test_R)
#29054個欠損がある変数は使用しない

test_R["AYB"] = np.where(test_R["AYB"].isnull(), test_R["AYB"].mode(), test_R["AYB"])

test_R["YR_RMDL"] = np.where(test_R["YR_RMDL"].isnull(), test_R["AYB"], test_R["YR_RMDL"]) #リモデルしてない物件は建築年で埋める)

test_R["QUADRANT"] = np.where(test_R["QUADRANT"].isnull(), test_R["QUADRANT"].mode(), test_R["QUADRANT"])

test_R["STORIES"] = np.where(test_R["STORIES"].isnull(), test_R["STORIES"].mode(), test_R["STORIES"])

test_R["KITCHENS"] = np.where(test_R["KITCHENS"].isnull(), test_R["KITCHENS"].mode(), test_R["KITCHENS"])

#(trainでは欠損じゃなかったキッチンが欠損なのがツライ気持ち)



#使用しない変数を落とす(Condominiamのみの変数)

test_R = test_R.drop(["LIVING_GBA", "CMPLX_NUM"], axis=1)



#使用しない変数を落とす（意味不明なもの）

test_R = test_R.drop(["Id", "SALE_NUM", "BLDG_NUM", "USECODE", "SOURCE"], axis=1)
#数値変数

test_R["HF_BATHRM2"] =  np.where(test_R["HF_BATHRM"] >= 3, 3, test_R["HF_BATHRM"]) 

test_R["NEW_total_BATHRM"] =  test_R["BATHRM"] + test_R["HF_BATHRM"]

test_R["ROOMS2"] = np.where(test_R["ROOMS"] >= 10, 10, test_R["ROOMS"])

test_R["NEW_BED/ROOMS"] = np.where(test_R["ROOMS2"]!=0, test_R["BEDRM"] / test_R["ROOMS2"], 0)

test_R["AYB2"] = np.abs(1950 - test_R["AYB"])

test_R["YR_RMDL2"] = np.where(test_R["YR_RMDL"] == 20, test_R["AYB2"], test_R["YR_RMDL"])

test_R["STORIES2"] = np.where(test_R["STORIES"] > 2, 2, test_R["STORIES"])

test_R["NEW_LAND/GBA"] = test_R["LANDAREA"] / test_R["GBA"]

test_R["NEW_AYB_to_RMDL"] = test_R["YR_RMDL2"] - test_R["AYB2"]

test_R["NEW_AYB_to_EYB"] = test_R["EYB"] - test_R["AYB2"]

test_R["NEW_EYB_to_YR_RMDL"] = np.abs(np.abs(test_R["YR_RMDL2"] - test_R["EYB"]) - 60)

test_R["KITCHENS2"] = np.where(test_R["KITCHENS"] > 5, 5, test_R["KITCHENS"])

test_R["KITCHENS3"] = np.abs(test_R["KITCHENS2"] - 3) ** 2

test_R["FIREPLACES2"] = np.where(test_R["FIREPLACES"] > 5, 5, test_R["FIREPLACES"])

test_R["SQUARE"] = np.where(test_R["SQUARE"] == "PAR ", test_R["SQUARE"].mode(), test_R["SQUARE"])

test_R["SQUARE"] = test_R["SQUARE"].astype("int") #"PAR "を置換してから数値にする

test_R["SALEDATE_Y"] = pd.to_datetime(test_R["SALEDATE"]).dt.year

#追加分

test_R["NEW_RMDL_to_SALE"] = test_R["SALEDATE_Y"] - test_R["YR_RMDL2"]

test_R["NEW_SQUARE/GBA"] = np.where(test_R["GBA"]==0, 0, test_R["SQUARE"] / test_R["GBA"])

test_R["NEW_BATH/ROOMS"] = np.where(test_R["ROOMS2"]==0, 0, test_R["NEW_total_BATHRM"] / test_R["ROOMS2"])





#カテゴリ変数

#(HEAT, STYLE、STRUCT,EXTWALL,ROOF,INTWALL(cadinaryが多いやつ)はあとで考える)

test_R["AC2"] = np.where(test_R["AC"]=="0", "N", test_R["AC"])

test_R["dum_AC_Y"] = np.where(test_R["AC2"]=="Y", 1, 0)

test_R["dum_QUALIFIED_Q"] = np.where(test_R["QUALIFIED"]=="Q", 1, 0)

test_R["dum_WARD_1"] = np.where(test_R["WARD"]=="Ward 1", 1, 0)

test_R["dum_WARD_2"] = np.where(test_R["WARD"]=="Ward 2", 1, 0)

test_R["dum_WARD_3"] = np.where(test_R["WARD"]=="Ward 3", 1, 0)

test_R["dum_WARD_4"] = np.where(test_R["WARD"]=="Ward 4", 1, 0)

test_R["dum_WARD_5"] = np.where(test_R["WARD"]=="Ward 5", 1, 0)

test_R["dum_WARD_6"] = np.where(test_R["WARD"]=="Ward 6", 1, 0)

test_R["dum_WARD_7"] = np.where(test_R["WARD"]=="Ward 7", 1, 0)

test_R["dum_QUADRANT_NW"] = np.where(test_R["QUADRANT"]=="NW", 1, 0)

test_R["dum_QUADRANT_SE"] = np.where(test_R["QUADRANT"]=="SE", 1, 0)

test_R["dum_QUADRANT_NE"] = np.where(test_R["QUADRANT"]=="NE", 1, 0)

test_R["GRADE"] = test_R["GRADE"].apply(GRADE_to_num)

test_R["CNDTN"] = test_R["CNDTN"].apply(CNDTN_to_num)



#変換元の変数は落とす

test_R = test_R.drop(["HF_BATHRM", "ROOMS", "AYB", "YR_RMDL", "FIREPLACES", "SALEDATE", "AC", "AC2", "QUALIFIED", "WARD", "QUADRANT", "HEAT"], axis=1)

test_R = test_R.drop(["STYLE", "STRUCT", "EXTWALL", "INTWALL", "ROOF"], axis=1)
#相関が高い変数を落とす

test_R = test_R.drop(["NUM_UNITS", "KITCHENS", "KITCHENS2", "YR_RMDL2"], axis=1)
pred_R_XGB = clf_R.predict(test_R)
test_C_Id = test[test["SOURCE"] =="Condominium"]["Id"]

test_R_Id= test[test["SOURCE"] =="Residential"]["Id"]

test_C_Id.reset_index(drop=True, inplace=True) 



pred_C_XGB2 = pd.concat([test_C_Id, pd.Series(pred_C_XGB)], axis=1, ignore_index=True)

pred_C_XGB2 = pred_C_XGB2.rename(columns={0: "Id", 1:"PRICE_log"})



pred_R_XGB2 = pd.concat([test_R_Id, pd.Series(pred_R_XGB)], axis=1, ignore_index=True)

pred_R_XGB2 = pred_R_XGB2.rename(columns={0: "Id", 1:"PRICE_log"})

print(pred_C_XGB2.shape)

print(pred_R_XGB2.shape)



submit2 = pd.concat([pred_C_XGB2, pred_R_XGB2], axis=0, ignore_index=True)

submit2["PRICE"] = np.exp(submit2["PRICE_log"]).astype("float32")

submit2 = submit2.drop("PRICE_log", axis=1)

submit2 = submit2.sort_values("Id")

print(submit2.shape)
##出力

submit2.to_csv("submit_20190123_1.csv", header=True, index=False)