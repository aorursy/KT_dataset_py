# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

%matplotlib inline

# 小数第4位まで表示
%precision 4
!pwd
features = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
stores = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')
test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
holidays = pd.read_csv('../input/holidays/holidays.csv')
# 型の変換
holidays["Date"]=pd.to_datetime(holidays["Date"])
features["IsHoliday"] = np.where(features["IsHoliday"]==True,1,0)
features["Date"] = pd.to_datetime(features["Date"])
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])
train["category"] = "train"
test["category"] = "test"
data_join = pd.concat([train,test])
data_join = pd.merge(data_join, features, on=["Store","Date", "IsHoliday"], how= "left")
data_join = pd.merge(data_join, stores, on="Store", how= "inner")
data_join = pd.merge(data_join, holidays, on="Date", how= "left")
data_join["IsHoliday"] = data_join["IsHoliday"].astype(int)
data_join.shape
data_join["YEAR"] = data_join["Date"].dt.year
data_join["MONTH"] = data_join["Date"].dt.month
data_join["WEEK"] = data_join["Date"].dt.week
data_join
# 特徴量作成：一年前の売上
data_sales = data_join[["Dept","Store","YEAR","WEEK","Weekly_Sales"]]
data_sales = data_sales.rename(columns={"Weekly_Sales":"Weekly_Sale_before"})
data_sales["YEAR"] = data_sales["YEAR"] + 1
data_join2 = pd.merge(data_join, data_sales, on=["Dept","Store","YEAR","WEEK"], how="left")
data_join2
# 特徴量作成：Dept72 & WEEK47 のフラグを立てる
data_join2["FLAG_D72&W47"] = np.where((data_join2["Dept"]==72) & (data_join2["WEEK"]==47),1,0)
data_join2[data_join2["FLAG_D72&W47"] == 1]
data_join2.isnull().sum()
data_join2["Weekly_Sale_before"] = data_join2["Weekly_Sale_before"].fillna(0)
data_join2["Weekly_Sale_before"] = np.where(data_join2["Weekly_Sale_before"]==0,data_join2["Weekly_Sales"],data_join2["Weekly_Sale_before"])
data_join2
data_join2.drop(columns=["Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","CPI","Unemployment"],axis=1,inplace=True)
data_join2
data_join2.loc[(data_join2.YEAR==2010) & (data_join2.WEEK==13), 'IsHoliday'] = True
data_join2.loc[(data_join2.YEAR==2011) & (data_join2.WEEK==16), 'IsHoliday'] = True
data_join2.loc[(data_join2.YEAR==2012) & (data_join2.WEEK==14), 'IsHoliday'] = True
data_join2.loc[(data_join2.YEAR==2013) & (data_join2.WEEK==13), 'IsHoliday'] = True
# data_join2["Weekly_Sales"] = np.where(data_join2["IsHoliday"]==1,data_join2["Weekly_Sales"]*5,data_join2["Weekly_Sales"])
# data_join2["Weekly_Sale_before"] = np.where(data_join2["IsHoliday"]==1,data_join2["Weekly_Sale_before"]*5,data_join2["Weekly_Sale_before"])
import lightgbm as lgb

# 参考HP
# https://rin-effort.com/2019/12/29/machine-learning-6/
# https://www.codexa.net/lightgbm-beginner/
data_train = data_join2[data_join2["category"]=="train"]
data_test = data_join2[data_join2["category"]=="test"]

train_x = data_train.drop(columns=['Date', 'Weekly_Sales', 'category'])
train_y = data_train['Weekly_Sales']
test_x = data_test.drop(columns=['Date', 'Weekly_Sales', 'category'])
train_x.columns
test_x.columns
obj_col = [col for col in train_x.columns if train_x[col].dtype == 'O']
obj_col
train_x[obj_col].head()
from sklearn.preprocessing import LabelEncoder

for c in obj_col:
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))
train_x[obj_col].head()
# import itertools

# param_space = {
#     'max_depth': [8, 12, 16],
#     "num_leaves": [8, 12, 16],
#     "min_data_in_leaf":[2, 6,10] 
# }

# param_combinations = itertools.product(param_space['max_depth'], param_space['num_leaves'],param_space['min_data_in_leaf'])

# for i,j,k in param_combinations:
#     print(f'max_depth:{i}, num_leaves:{j}, min_data_in_leaf:{k}')
# #イテレータは一度for文を回すと再実行しても最初から回ってくれない。上でfor文を回してしまったので、改めてイテレータを発行する。
# param_combinations = itertools.product(param_space['max_depth'], param_space['num_leaves'],param_space['min_data_in_leaf'])

# #•パラメータの組ごとにそのパラメータとスコアを保存するリストを用意
# params_list = []
# scores_list = []
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# version13 追記　：　MAE
from sklearn.metrics import mean_absolute_error
# %%time
# for max_depth,num_leaves,min_data_in_leaf in param_combinations:
#     kf = KFold(n_splits=3, shuffle=True, random_state=72)
#     train_scores = []
#     valid_scores = []
#     print(f'param: max_depth={max_depth}, num_leaves={num_leaves},min_data_in_leaf={min_data_in_leaf} ')

#     # ハイパーパラメータ
#     params = {"metric": "rmse",
#               "max_depth" : max_depth,
#               "num_leaves": num_leaves,
#               "min_data_in_leaf":min_data_in_leaf}
    
#     for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
#         tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
#         tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

#         lgb_train = lgb.Dataset(tr_x, tr_y)
#         lgb_eval = lgb.Dataset(va_x, va_y)

#         gbm = lgb.train(params,lgb_train,valid_sets=lgb_eval,num_boost_round=1000,early_stopping_rounds=100,verbose_eval=500)

#         tr_pred = gbm.predict(tr_x)
#         va_pred = gbm.predict(va_x)

#         train_RMSE = np.sqrt(mean_squared_error(tr_y,tr_pred))
#         valid_RMSE = np.sqrt(mean_squared_error(va_y,va_pred))

#         train_scores.append(train_RMSE)
#         valid_scores.append(valid_RMSE)

#         print(f'fold{i}:  train_RMSE={train_RMSE}  valid_RMSE={valid_RMSE}')

#     mean_train_scores = np.mean(train_scores)
#     mean_valid_scores = np.mean(valid_scores)
#     params_list.append((max_depth,num_leaves,min_data_in_leaf))
#     scores_list.append(mean_valid_scores)
# best_idx = np.argsort(scores_list)[-1]
# best_param = params_list[best_idx]
# print(f'best param: max_depth={best_param[0]}, num_leaves={best_param[1]},min_data_in_leaf={best_param[2]}')
# best param: max_depth=16, num_leaves=8,min_data_in_leaf=2
best_param = [16,8,2]
# ハイパーパラメータ
# チューニングで決定したbest_paramの値を代入
params = {"metric": "mae",
          "max_depth":best_param[0],
          "num_leaves":best_param[1],
          "min_data_in_leaf":best_param[2]
         }

# 初期値
train_scores = []
valid_scores = []
pred = np.zeros(test_x.shape[0])
pred_valid = np.zeros(train_x.shape[0])

# 5kfold クロスバリデーション

kf = KFold(n_splits = 5, shuffle = True, random_state=0)

for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(va_x, va_y)

    gbm = lgb.train(params,lgb_train,valid_sets=lgb_eval,num_boost_round=10000,early_stopping_rounds=100,verbose_eval=500)
    
    tr_pred = gbm.predict(tr_x)
    va_pred = gbm.predict(va_x)
    pred += gbm.predict(test_x) / 5
    
    # version9 追記 各varid時に予測値を求めておく
    pred_valid[va_idx] =  va_pred

    train_MAE = mean_absolute_error(tr_y,tr_pred)
    valid_MAE = mean_absolute_error(va_y,va_pred)

    train_scores.append(train_MAE)
    valid_scores.append(valid_MAE)
                       
    print(f'fold{i}:  train_MAE={train_MAE}  valid_MAE={valid_MAE}')

mean_train_scores = np.mean(train_scores)
mean_valid_scores = np.mean(valid_scores)

print(f'mean train MAE={mean_train_scores}  mean valid MAE={mean_valid_scores}')
#予測値と正解値を描写する関数
def True_Pred_map(true,pred):
    R2 = r2_score(true, pred) 
    plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    ax.scatter(x=true,y=pred)
    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Pred Value', fontsize=15)
    ax.set_xlim(true.min()-0.1 , true.max()+0.1)
    ax.set_ylim(pred.min()-0.1 , pred.max()+0.1)
    x = np.linspace(true.min()-0.1, true.max()+0.1, 2)
    y = x
    ax.plot(x,y,'r-')
    plt.text(0.1, 0.8, 'R^2 = {}'.format(str(round(R2, 5))), transform=ax.transAxes, fontsize=15)
True_Pred_map(tr_y,tr_pred)
True_Pred_map(va_y,va_pred)
# Feature_importance
lgb.plot_importance(gbm, importance_type="gain",height=0.5, figsize=(8,6))
# from graphviz import Digraph
# lgb.create_tree_digraph(gbm) # figsize=(20,12)
train_tmp = train_x.copy()
train_tmp["Weekly_Sales"] = train_y
train_tmp["pred"] = pred_valid
train_tmp["pred_diff"] = train_tmp['Weekly_Sales']-train_tmp["pred"]
train_tmp
train_tmp["pred_diff"].hist(bins=50).set_yscale("log")
train_tmp[(train_tmp["pred_diff"]<-200000) | (train_tmp["pred_diff"]>200000)]
# trainとtestの各Deptのレコード数
tmp = data_join2.pivot_table(index="Dept",columns="category",values="Type",aggfunc="count")
tmp.plot(kind="bar",figsize=(20,5))
# trainとtestの各Storeのレコード数
tmp = data_join2.pivot_table(index="Store",columns="category",values="Type",aggfunc="count")
tmp.plot(kind="bar",figsize=(20,5))
# 時系列的に並べてみる
plt.figure(figsize=(20,5))
sns.lineplot(x="WEEK", y="Weekly_Sales", data=data_join2[data_join2["Dept"]==72],hue="YEAR")
# Week47の各deptでの反応の違い
plt.figure(figsize=(20,5))
sns.barplot(x="Dept", y="Weekly_Sales", data=data_join2[(data_join2["WEEK"]==46) | (data_join2["WEEK"]==47)],hue="WEEK")
# WEEKとHolidayが年によってずれていってないか？
data_join2[(data_join2["Store"] == 1) & (data_join2["Dept"] == 1) & (data_join2["Holiday_name"].notnull())]
# Store毎のDept72 & WEEK47　の反応の違い
plt.figure(figsize=(20,5))
sns.barplot(x="Store", y="Weekly_Sales", data=data_join2[(data_join2["Dept"]==72) & ((data_join2["WEEK"]==46) | (data_join2["WEEK"]==47))],hue="WEEK")
sns.scatterplot(data = data_join2[((data_join2["Dept"]==72)|(data_join2["Dept"]==92)) & (data_join2["WEEK"]==47)],x="Weekly_Sale_before",y="Weekly_Sales",hue="YEAR")
# もう一度、ズレ量の各パラメータ依存性を見返してみる
fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(20, 15))
sns.scatterplot(data=train_tmp,x="Store",y="pred_diff",hue="YEAR",ax=ax1,palette="Set2")
sns.scatterplot(data=train_tmp,x="Dept",y="pred_diff",hue="YEAR",ax=ax2,palette="Set2")
sns.scatterplot(data=train_tmp,x="WEEK",y="pred_diff",hue="YEAR",ax=ax3,palette="Set2")
fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(20, 15))
sns.scatterplot(data=train_tmp,x="Store",y="pred_diff",hue="Holiday_name",ax=ax1,palette="Set2")
sns.scatterplot(data=train_tmp,x="Dept",y="pred_diff",hue="Holiday_name",ax=ax2,palette="Set2")
sns.scatterplot(data=train_tmp,x="YEAR",y="pred_diff",hue="Holiday_name",ax=ax3,palette="Set2")
plt.figure(figsize=(15,15))
sns.scatterplot(data = train_tmp, x="Weekly_Sale_before",y="pred_diff", hue="Holiday_name",palette="Set2")
fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(20, 15))
sns.barplot(data=train_tmp[(train_tmp["Dept"]==72) & (train_tmp["WEEK"]==47)],x="Store",y="pred_diff",hue="YEAR",ax=ax1,palette="Set2")
sns.barplot(data=train_tmp[(train_tmp["Dept"]==72) & (train_tmp["WEEK"]==47)],x="Store",y="Weekly_Sales",hue="YEAR",ax=ax2,palette="Set2")
sns.barplot(data=train_tmp[(train_tmp["Dept"]==72) & (train_tmp["WEEK"]==47)],x="Store",y="Weekly_Sale_before",hue="YEAR",ax=ax3,palette="Set2")
test_pred = test.copy(deep=True)
test_pred["Weekly_Sales"] = pred
test_pred[['Store', 'Dept', 'Date']] = test_pred[['Store', 'Dept', 'Date']].astype(str)
test_pred["id"] = test_pred["Store"] + "_" + test_pred["Dept"] + "_" + test_pred["Date"]

# test_pred = test_pred[["id","Weekly_Sales"]]
# test_pred.head()
# YEARとWEEKをtestデータで再度作る
test_pred["Date"] = pd.to_datetime(test_pred["Date"])
test_pred["YEAR"] = test_pred["Date"].dt.year
test_pred["MONTH"] = test_pred["Date"].dt.month
test_pred["WEEK"] = test_pred["Date"].dt.week
test_pred
# Version15更新：Dept72 & WEEK47は、一年前の売上で置換
# 先に作っていた 1年前の売上が入ったdata_salesをマージして、置換

# 先程idづくりのためにstrにしていたのでintに戻す
test_pred[['Store', 'Dept']] = test_pred[['Store', 'Dept']].astype(int)

test_pred_bef = pd.merge(test_pred, data_sales, on=["Dept","Store","YEAR","WEEK"], how="left")
test_pred_bef[(test_pred_bef["Dept"] == 72) & (test_pred_bef["WEEK"] == 47)]
test_pred_bef["Weekly_Sales"] = np.where((test_pred_bef["Dept"] == 72) & (test_pred_bef["WEEK"] == 47),  test_pred_bef["Weekly_Sale_before"], test_pred_bef["Weekly_Sales"])
test_pred_bef[(test_pred_bef["Dept"] == 72) & (test_pred_bef["WEEK"] == 47)]
test_pred2 = test_pred_bef.copy()
test_pred2["Weekly_Sales"] = np.where((test_pred2["YEAR"]==2012)&(test_pred2["WEEK"]==52)&(test_pred2["Weekly_Sales"].shift(1) > 2*test_pred2["Weekly_Sales"])\
                                     ,test_pred2["Weekly_Sales"]+test_pred2["Weekly_Sales"].shift(1)*2.5/7, test_pred2["Weekly_Sales"])
# test_pred2["Weekly_Sales"] = np.where(test_pred2["IsHoliday"]==1,test_pred2["Weekly_Sales"]/5,test_pred2["Weekly_Sales"])
test_pred3 = test_pred2[["id","Weekly_Sales"]]
test_pred3.head()
submission = test_pred3[["id","Weekly_Sales"]]
submission.to_csv('submission15.csv', index=False)