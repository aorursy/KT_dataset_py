import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error, r2_score

from math import sqrt



from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.neighbors import KNeighborsRegressor

import lightgbm as lgbm
train = pd.read_csv("/kaggle/input/restaurant-revenue-prediction/train.csv.zip")

test = pd.read_csv("/kaggle/input/restaurant-revenue-prediction/test.csv.zip")

ref = pd.read_csv("/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv")

test_id = pd.DataFrame(test["Id"])

display(ref.head())
a = train.isnull().sum()

a[a>0]
display(train.head(), test.head())
print(train["City"].groupby(train["City"]).count().count())

print(train["City Group"].groupby(train["City Group"]).count().count())

print(train["Type"].groupby(train["Type"]).count().count())

print(test["City"].groupby(test["City"]).count().count())

print(test["City Group"].groupby(test["City Group"]).count().count())

print(test["Type"].groupby(test["Type"]).count().count())
train["kOpen Year"] = train["Open Date"].str[6:12]

test["kOpen Year"] = test["Open Date"].str[6:12]

train["Open Month"] = train["Open Date"].str[0:2].astype(int)

test["Open Month"] = test["Open Date"].str[0:2].astype(int)

train["gOpen Year's"] = train["Open Date"].str[8:9] + "0's"

test["gOpen Year's"] = test["Open Date"].str[8:9] + "0's"

train['Open Month'] = pd.cut(train['Open Month'],4,labels=['1','2','3','4'])

test['Open Month'] = pd.cut(test['Open Month'],4,labels=['1','2','3','4'])

# train, test = train.drop(["Id","Open Date"], axis=1), test.drop(["Id","Open Date"], axis=1)

train, test = train.drop(["Id","Open Date","City"], axis=1), test.drop(["Id","Open Date","City"], axis=1)
train["gOpen Year's"].groupby(train["gOpen Year's"]).count()
train.head()
train.loc[:,"P1":"P37"].describe()
corr_abs = np.absolute(train.loc[:,"P1":"revenue"].corr()["revenue"].drop("revenue")).sort_values(ascending=False)   

ex_var = pd.DataFrame(corr_abs[corr_abs<0.1]).index

train1 = train.drop(ex_var, axis=1)

test1 = test.drop(ex_var, axis=1)
# train_dm = pd.get_dummies(train1)

# test_dm = pd.get_dummies(test1)

train_dm = pd.get_dummies(train).drop("City Group_Other", axis=1)

test_dm = pd.get_dummies(test).drop("City Group_Other", axis=1)

print(train_dm.shape, test_dm.shape)

print(train_dm.columns,"\n", test_dm.columns)
a = [x for x in train_dm.columns]+ [y for y in test_dm.columns]

a = set(a)

print(a, len(a))
train_fix = pd.DataFrame(train_dm, columns=a)

test_fix = pd.DataFrame(test_dm, columns=a) 
train_fix = train_fix.fillna(0)

test_fix = test_fix.fillna(0)
X, y = train_fix.drop(["revenue"], axis=1), train_dm["revenue"]

test_fix = test_fix.drop("revenue", axis=1)
print(train_fix.shape,test_fix.shape)
kX = X.drop(X.loc[:,X.columns.str.startswith('gOpen')], axis=1)

gX = X.drop(X.loc[:,X.columns.str.startswith('kOpen')], axis=1)

ktest_fix = test_fix.drop(test_fix.loc[:,test_fix.columns.str.startswith('gOpen')], axis=1)

gtest_fix = test_fix.drop(test_fix.loc[:,test_fix.columns.str.startswith('kOpen')], axis=1)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)



# kX_train, kX_valid, ky_train, ky_valid = train_test_split(X.drop(X.loc[:,X.columns.str.startswith('gOpen')].columns, axis=1), y, test_size=0.3, shuffle=True, random_state=42)

# gX_train, gX_valid, gy_train, gy_valid = train_test_split(X.drop(X.loc[:,X.columns.str.startswith('kOpen')].columns, axis=1), y, test_size=0.3, shuffle=True, random_state=42)

# print(kX_train.shape, ky_train.shape, kX_valid.shape, ky_valid.shape)

# print(gX_train.shape, gy_train.shape, gX_valid.shape, gy_valid.shape)
gbm1 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, criterion='mse', random_state=42)

gb0 = GradientBoostingRegressor(n_estimators=150, learning_rate=0.01, criterion='mse', random_state=42)

gb1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, criterion='mse', random_state=42)

gb2 = GradientBoostingRegressor(n_estimators=75, learning_rate=0.005, criterion='mse', random_state=42)

gb3 = GradientBoostingRegressor(n_estimators=50, learning_rate=0.005, criterion='mse', random_state=42)

# xgb1 = XGBRegressor(n_estimators=100, learning_rate=0.01, criterion='mse', random_state=42)

# rf1 = RandomForestRegressor(n_estimators=125, criterion='mse', random_state=42)

# rf2 = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=42)

# rf3 = RandomForestRegressor(n_estimators=75, criterion='mse', random_state=42)

knn = KNeighborsRegressor()

# knn1 = KNeighborsRegressor(n_neighbors=10)
def rmse(mod):

    pred_tr = mod.predict(X_train)

    pred_val = mod.predict(X_valid)

    rmse_tr = round(sqrt(mean_squared_error(pred_tr, y_train)), 4)

    rmse_val = round(sqrt(mean_squared_error(pred_val, y_valid)), 4)

    print(f"学習データに対するスコア:{rmse_tr}\n検証データに対するスコア:{rmse_val}")
def cross_rmse(mod, X):

    tr_scores = cross_val_score(mod, X, y, scoring='neg_root_mean_squared_error',cv=10)

    val_scores = cross_val_score(mod, X, y, scoring='neg_root_mean_squared_error', cv=10)

    val_r2 = cross_val_score(mod, X, y, scoring='r2', cv=10)

#     print(f"学習データに対するスコア:{-tr_scores}\n検証データに対するスコア:{-val_scores}")

    print(f"学習に対する平均スコア:{-np.mean(tr_scores)}\n検証データに対する平均スコア:{-np.mean(val_scores)}")

    print(f"学習に対するスコアの分散:{np.var(-tr_scores)}\n検証に対するスコアの分散:{np.var(-val_scores)}")

    print(val_r2)

    print(np.mean(val_r2))
# rmse(gb1)

# rmse(rf1)

cross_rmse(gbm1, gX)

print("--------------------------------------------")

cross_rmse(gb0, gX)

print("--------------------------------------------")

# cross_rmse(gb1, gX)

# print("--------------------------------------------")

# cross_rmse(gb2, gX)

# print("--------------------------------------------")

cross_rmse(gb3, gX)

print("--------------------------------------------")

# cross_rmse(rf1, kX)

# print("--------------------------------------------")

# cross_rmse(rf2, kX)

# print("--------------------------------------------")

# cross_rmse(rf3, kX)

# print("--------------------------------------------")

cross_rmse(knn, kX)
gbm1.fit(gX, y)

gb0.fit(gX, y)

gb1.fit(gX, y)

gb2.fit(gX, y)

gb3.fit(gX, y)

# xgb1.fit(gX, y)

# rf1.fit(kX, y)

# rf2.fit(kX, y)

# rf3.fit(kX, y)

knn.fit(kX, y)

# knn1.fit(kX, y)
# 学習に対する平均スコア:2417363.8602510625

# 検証データに対する平均スコア:2417363.8602510625

# 学習に対するスコアの分散:994983448212.0974

# 検証に対するスコアの分散:994983448212.0974

# [ 0.22959198 -0.02355106 -0.04131754  0.02320255 -1.42121393 -0.01339423

#  -0.50032525 -0.00511683 -0.94482666 -0.09574994]

# -0.27927009048416584

# --------------------------------------------

# 学習に対する平均スコア:2262724.311816029

# 検証データに対する平均スコア:2262724.311816029

# 学習に対するスコアの分散:1034304961972.9343

# 検証に対するスコアの分散:1034304961972.9343

# [ 0.6758959   0.10599187 -0.33771629  0.06035573 -0.70060848 -0.06566423

#  -0.22722636  0.00479488 -0.46487618  0.11093334]

# -0.0838119815005104

# --------------------------------------------

# 学習に対する平均スコア:2255198.463811924

# 検証データに対する平均スコア:2255198.463811924

# 学習に対するスコアの分散:1026121013795.4535

# 検証に対するスコアの分散:1026121013795.4535

# [ 0.66427828  0.10048079 -0.38268166  0.0759842  -0.79612978 -0.0764769

#  -0.16552763  0.03784851 -0.4053034   0.12401337]

# -0.08235142187160203

# --------------------------------------------

# 学習に対する平均スコア:2267515.4590375493

# 検証データに対する平均スコア:2267515.4590375493

# 学習に対するスコアの分散:1022303386669.3235

# 検証に対するスコアの分散:1022303386669.3235

# [ 0.63151664  0.10080647 -0.33594766  0.04437104 -0.70846024 -0.07733577

#  -0.3096063   0.03866633 -0.34245722  0.09824694]

# -0.08601997539116511

# --------------------------------------------

# 学習に対する平均スコア:2363682.054061462

# 検証データに対する平均スコア:2363682.054061462

# 学習に対するスコアの分散:1033851699981.6838

# 検証に対するスコアの分散:1033851699981.6838

# [ 0.04817628 -0.0024224  -0.78942141 -0.22785307 -0.53783367  0.00676171

#   0.36296991 -0.16077062 -0.24970429 -0.28928497]

# -0.18393825266078892
# 学習に対する平均スコア:2596059.604957393

# 検証データに対する平均スコア:2267029.693829722

# 学習に対するスコアの分散:1794149015394.774

# 検証に対するスコアの分散:780409314628.8933

# --------------------------------------------

# 学習に対する平均スコア:2380533.6119319233

# 検証データに対する平均スコア:2236708.0720026055

# 学習に対するスコアの分散:1523180467170.3706

# 検証に対するスコアの分散:887241452395.4861

# --------------------------------------------

# 学習に対する平均スコア:2327755.614798424

# 検証データに対する平均スコア:2269469.4081999627

# 学習に対するスコアの分散:1488451397557.3389

# 検証に対するスコアの分散:923778585009.8998
def submit(mod, test_fix):

    pred = pd.DataFrame(mod.predict(test_fix))

    submit = pd.concat([test_id, pred], axis=1).rename(columns={0: 'Prediction'})

    return submit
def ensemble_submit(mod1, mod2, rate1):

    pred = pd.DataFrame(mod1.predict(gtest_fix)*rate1 + mod2.predict(ktest_fix)*(1-rate1))

    submit = pd.concat([test_id, pred], axis=1).rename(columns={0: 'Prediction'})

    return submit

def ensemble_submit3(mod1, mod2, mod3, rate):

    pred = pd.DataFrame(mod1.predict(gtest_fix)*rate[0] + mod2.predict(ktest_fix)*rate[1] + mod3.predict(gtest_fix)*rate[2])

    submit = pd.concat([test_id, pred], axis=1).rename(columns={0: 'Prediction'})

    return submit    

def ensemble_submitx3(mod1, mod2, mod3):

    pred = pd.DataFrame((mod1.predict(gtest_fix) * mod2.predict(ktest_fix) * mod3.predict(gtest_fix))**(1/3))

    submit = pd.concat([test_id, pred], axis=1).rename(columns={0: 'Prediction'})

    return submit    

from sklearn.neural_network import MLPRegressor

# tmp_X = pd.DataFrame(gb1.predict(gX)*0.7 + knn.predict(kX)*0.3)

nn = MLPRegressor(random_state=42)

# nn.fit(tmp_X, y)

nn.fit(gX, y)
gb0_submit = submit(gb0, gtest_fix)

gb1_submit = submit(gb1, gtest_fix)

gb2_submit = submit(gb2, gtest_fix)

gb3_submit = submit(gb3, gtest_fix)

# xgb1_submit = submit(xgb1, gtest_fix)

# rf1_submit = submit(rf1, ktest_fix)

# rf2_submit = submit(rf2, ktest_fix)

# rf3_submit = submit(rf3, ktest_fix)

knn_submit = submit(knn, ktest_fix)



# gb7_kn3_submit = ensemble_submit(gb1, knn, 0.7)

# gb5_kn5_submit = ensemble_submit(gb1, knn, 0.5)

# gb4_kn6_submit = ensemble_submit(gb1, knn, 0.4)

# gb3_kn7_submit = ensemble_submit(gb1, knn, 0.3)

# gb7_kn3_submit = ensemble_submit(gb1, knn, 0.7)

# gb8_kn2_submit = ensemble_submit(gb1, knn, 0.8)



# gb4_kn6_submit2 = ensemble_submit(gb2, knn, 0.4)

# gb7_kn3_submit2 = ensemble_submit(gb2, knn, 0.7)





gb4_kn6_submit3 = ensemble_submit(gb3, knn, 0.4)

# gb7_kn3_submit3 = ensemble_submit(gb3, knn, 0.7)



# gb4_kn6_submit0 = ensemble_submit(gb0, knn, 0.4)

# gb7_kn3_submit0 = ensemble_submit(gb0, knn, 0.7)



# gb4_kn6_submitm1 = ensemble_submit(gbm1, knn, 0.4)

# gb7_kn3_submitm1 = ensemble_submit(gbm1, knn, 0.7)





# xgb4_kn6_submit = ensemble_submit(xgb1, knn, 0.4)

# xgb7_kn3_submit = ensemble_submit(xgb1, knn, 0.7)



# gb2_kn7_submit_m1 = ensemble_submit3(gb0, knn, gbm1, [0.2, 0.7, 0.1])

# gb7_kn2_submit_m1 = ensemble_submit3(gb0, knn, gbm1, [0.7, 0.2, 0.1])



# gb2_kn7_submit3 = ensemble_submit3(gb0, knn, gb3, [0.2, 0.7, 0.1])

# gb8_kn1_submit3 = ensemble_submit3(gb0, knn, gb3, [0.8, 0.1, 0.1])

# gb7_kn2_submit3 = ensemble_submit3(gb0, knn, gb3, [0.7, 0.2, 0.1])

# gbkngb_622_submit = ensemble_submit3(gb0, knn, gb3, [0.6, 0.2, 0.2])

# gbkngb_622_submit1 = ensemble_submit3(gb0, knn1, gb3, [0.6, 0.2, 0.2])

gbkngb_55_submit = ensemble_submit3(gb0, knn, gb3, [0.5, 0.25, 0.25])

gbkngb_523_submit = ensemble_submit3(gb0, knn, gb3, [0.5, 0.2, 0.3])

gbkngb_433_submit = ensemble_submit3(gb0, knn, gb3, [0.4, 0.3, 0.3])



test55 = ensemble_submitx3(gb0, knn, gb3)



# gbkngb_55_submit1 = ensemble_submit3(gb0, knn1, gb3, [0.5, 0.25, 0.25])





# gb7_kn2_submit = ensemble_submit3(gb1, knn, rf3, [0.7, 0.2, 0.1])

# gb45_kn45_submit = ensemble_submit3(gb1, knn, rf3, [0.45, 0.45, 0.1])

# gb2_kn7_submit = ensemble_submit3(gb1, knn, rf3, [0.2, 0.7, 0.1])
# nn_pred = pd.DataFrame(nn.predict(pd.DataFrame(gb1.predict(gtest_fix)*0.7 + knn.predict(ktest_fix)*(1-0.7))))

# nn_submit = pd.concat([test_id, nn_pred], axis=1).rename(columns={0: 'Prediction'})
# gb1_submit.to_csv("gb1_submission.csv", index=False)

# rf1_submit.to_csv("rf1_submission.csv", index=False)

# rf2_submit.to_csv("rf2_submission.csv", index=False)

# rf3_submit.to_csv("rf3_submission.csv", index=False)

# knn_submit.to_csv("knn_submission.csv", index=False)



# gb7_kn3_submit.to_csv("gb7_kn3_submit.csv", index=False)

# gb5_kn5_submit.to_csv("gb5_kn5_submit.csv", index=False)

# gb4_kn6_submit.to_csv("gb4_kn6_submit.csv", index=False)

# gb3_kn7_submit.to_csv("gb3_kn7_submit.csv", index=False)

# gb7_kn3_submit.to_csv("gb7_kn3_submit.csv", index=False)

# gb8_kn2_submit.to_csv("gb8_kn2_submit.csv", index=False)



# gb4_kn6_submit2.to_csv("gb4_kn6_submit2.csv", index=False)

# gb7_kn3_submit2.to_csv("gb7_kn3_submit2.csv", index=False)



gb4_kn6_submit3.to_csv("gb4_kn6_submit3.csv", index=False)

# gb7_kn3_submit3.to_csv("gb7_kn3_submit3.csv", index=False)



# gb4_kn6_submit0.to_csv("gb4_kn6_submit0.csv", index=False)

# gb7_kn3_submit0.to_csv("gb7_kn3_submit0.csv", index=False)



# gb4_kn6_submitm1.to_csv("gb4_kn6_submitm1.csv", index=False)

# gb7_kn3_submitm1.to_csv("gb7_kn3_submitm1.csv", index=False)



# nn_submit.to_csv("nn_stack_73_submit.csv", index=False)



# xgb4_kn6_submit.to_csv("xgb4_kn6_submit.csv", index=False)

# xgb7_kn3_submit.to_csv("xgb7_kn3_submit.csv", index=False)



# gb7_kn2_submit_m1.to_csv("gb7_kn2_submit_m1.csv", index=False)

# gb2_kn7_submit_m1.to_csv("gb2_kn7_submit_m1.csv", index=False)



# gb8_kn1_submit3.to_csv("gb8_kn1_submit3.csv", index=False)

# gb7_kn2_submit3.to_csv("gb7_kn2_submit3.csv", index=False)

# gb2_kn7_submit3.to_csv("gb2_kn7_submit3.csv", index=False)



# gbkngb_622_submit.to_csv("gbkngb_622_submit.csv", index=False)

# gbkngb_622_submit1.to_csv("gbkngb_622_submit1.csv", index=False)

gbkngb_55_submit.to_csv("gbkngb_55_submit.csv", index=False)

# gbkngb_55_submit1.to_csv("gbkngb_55_submit1.csv", index=False)

gbkngb_523_submit.to_csv("gbkngb_523_submit.csv", index=False)

gbkngb_433_submit.to_csv("gbkngb_433_submit.csv", index=False)



test55.to_csv("test55.csv", index=False)



# gb7_kn2_submit.to_csv("gb7_kn2_submit.csv", index=False)

# gb45_kn45_submit.to_csv("gb45_kn45_submit.csv", index=False)

# gb2_kn7_submit.to_csv("gb2_kn7_submit.csv", index=False)

# submit.to_csv("submission.csv", index=False)
# plt.figure(figsize=(15,6))

# sns.countplot(train["kOpen Year"])

# sns.countplot(test["Open Year"])



# plt.figure(figsize=(15,6))

# sns.countplot(train["Open Month"])

# sns.countplot(test["Open Month"])