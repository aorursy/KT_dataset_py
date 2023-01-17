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
!ls /
! unzip /kaggle/input/recruit-restaurant-visitor-forecasting/hpg_store_info.csv.zip
! unzip /kaggle/input/recruit-restaurant-visitor-forecasting/hpg_reserve.csv.zip
! unzip /kaggle/input/recruit-restaurant-visitor-forecasting/air_visit_data.csv.zip
! unzip /kaggle/input/recruit-restaurant-visitor-forecasting/air_reserve.csv.zip
! unzip /kaggle/input/recruit-restaurant-visitor-forecasting/store_id_relation.csv.zip
! unzip /kaggle/input/recruit-restaurant-visitor-forecasting/air_store_info.csv.zip
! unzip /kaggle/input/recruit-restaurant-visitor-forecasting/sample_submission.csv.zip
! unzip /kaggle/input/recruit-restaurant-visitor-forecasting/date_info.csv.zip
import os
files = []
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        if "ipynb" in filename:
            continue
        files.append(os.path.join(dirname, filename))
files
df_ss= pd.read_csv('./sample_submission.csv')
df_hr= pd.read_csv('./hpg_reserve.csv')
df_asi= pd.read_csv('./air_store_info.csv')
df_ar= pd.read_csv('./air_reserve.csv')
df_avd= pd.read_csv('./air_visit_data.csv')
df_hsi= pd.read_csv('./hpg_store_info.csv')
df_sir= pd.read_csv('./store_id_relation.csv')
df_di= pd.read_csv('./date_info.csv')
# 予測用air_store_id_日付,visitors
df_ss.head()
air_store_ids=df_ss.id.str.split("_", expand = True)[0]+"_"+df_ss.id.str.split("_", expand = True)[1]
date=df_ss.id.str.split("_", expand = True)[2]
df_ss["air_store_id"]=air_store_ids
df_ss["visit_date"]=date
df_ss.head()
# 予約情報 hpg_store_id, 日付と時間 予約人数
df_hr.head()
#df_hr.count()
# 扱いづらいので展開しておく
df_hr["visit_date"] = df_hr.visit_datetime.str.split(expand = True)[0]
df_hr["reserve_date"]= df_hr.reserve_datetime.str.split(expand = True)[0]
df_hr_grouped=df_hr.groupby(['hpg_store_id', 'visit_date']).sum()
# air store info id, ジャンル, エリア, 緯度経度
df_asi.head()
# df_asi.count()
# 予約情報  air_store_id, 日付時間, 予約人数
df_ar.head()
# 扱いづらいので展開しておく
df_ar["visit_date"] = df_ar.visit_datetime.str.split(expand = True)[0]
df_ar["reserve_date"]= df_ar.reserve_datetime.str.split(expand = True)[0]
# こんな感じで使っておく
df_ar.groupby(['air_store_id', 'visit_date']).sum()
df_ar_grouped=df_ar.groupby(['air_store_id', 'visit_date']).sum()
# 訪問人数 (解答に近い形式) 
df_avd.head()
# 店舗情報, hpg_store_id, ジャンル, エリア, 緯度経度
df_hsi.head()
# air_store_idとhpg_store_idの対応表
df_sir.head()
# 日付の情報
df_di.head()
df_avd_merged = df_avd.merge(df_di, left_on='visit_date', right_on='calendar_date', how="left")
df_avd_merged = df_avd_merged.merge(df_sir, left_on='air_store_id', right_on='air_store_id', how="left")
df_avd_merged = df_avd_merged.merge(df_hsi, left_on='hpg_store_id', right_on='hpg_store_id', how="left")
df_avd_merged = df_avd_merged.merge(df_ar_grouped, left_on=['air_store_id', 'visit_date'], right_on=['air_store_id', 'visit_date'], how="left")
df_avd_merged = df_avd_merged.merge(df_hr_grouped, left_on=['hpg_store_id', 'visit_date'], right_on=['hpg_store_id', 'visit_date'], how="left")

df_ss_merged = df_ss.merge(df_di, left_on='visit_date', right_on='calendar_date', how="left")
df_ss_merged = df_ss_merged.merge(df_sir, left_on='air_store_id', right_on='air_store_id', how="left")
df_ss_merged = df_ss_merged.merge(df_hsi, left_on='hpg_store_id', right_on='hpg_store_id', how="left")
df_ss_merged = df_ss_merged.merge(df_ar_grouped, left_on=['air_store_id', 'visit_date'], right_on=['air_store_id', 'visit_date'], how="left")
df_ss_merged = df_ss_merged.merge(df_hr_grouped, left_on=['hpg_store_id', 'visit_date'], right_on=['hpg_store_id', 'visit_date'], how="left")

df_ss_merged.head()
# あとは昼休みに続き
# 日付からハイフンを取り除く
# df_avd_merged["visit_date"] = df_avd_merged["visit_date"].map(lambda v : v.replace("-", ""))
# df_ss_merged["visit_date"] = df_ss_merged["visit_date"].map(lambda v : v.replace("-", ""))

# 曜日情報を付与
df_avd_merged["week"] = df_avd_merged["visit_date"].map(lambda v : int(v.replace("-", ""))%7)
df_ss_merged["week"]= df_ss_merged["visit_date"].map(lambda v : int(v.replace("-", ""))%7)
# datetime = pd.to_datetime(df_avd_merged["visit_date"], format='%Y-%m-%d', errors='ignore')
# df_avd_merged["timestamp"] = (datetime - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
# datetime = pd.to_datetime(df_ss_merged["visit_date"], format='%Y-%m-%d', errors='ignore')
# df_ss_merged["timestamp"] = (datetime - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
# 移動平均を設定　テストデータは末尾のデータ
df_avd_merged["moving_ave"] = df_avd_merged["visitors"].rolling(window=3, min_periods=1).mean()
df_ss_merged["moving_ave"] = df_avd_merged.groupby(["air_store_id"]).last().moving_ave
# air_store_idごとの平均値を計算
mean_visitors=df_avd_merged.groupby(["air_store_id"]).mean()
mean_visitors=pd.DataFrame(data=mean_visitors, columns=["visitors"])
mean_visitors["mean_visitors"] = mean_visitors["visitors"]
mean_visitors=pd.DataFrame(data=mean_visitors, columns=["mean_visitors"])
# 平均値を付与しておく
df_avd_merged = df_avd_merged.merge(mean_visitors, left_on='air_store_id', right_on='air_store_id', how="left")
df_ss_merged = df_ss_merged.merge(mean_visitors, left_on='air_store_id', right_on='air_store_id', how="left")
# air_store_id 曜日ごとの平均値を計算
mean_visitors_w=df_avd_merged.groupby(["air_store_id", "week"]).mean()
mean_visitors_w=pd.DataFrame(data=mean_visitors_w, columns=["visitors"])
mean_visitors_w["mean_visitors_w"] = mean_visitors_w["visitors"]
mean_visitors_w=pd.DataFrame(data=mean_visitors_w, columns=["mean_visitors_w"])
# 曜日ごとの平均値を付与しておく
df_avd_merged = df_avd_merged.merge(mean_visitors_w, left_on=["air_store_id", "week"], right_on=["air_store_id", "week"], how="left")
df_ss_merged = df_ss_merged.merge(mean_visitors_w, left_on=["air_store_id", "week"], right_on=["air_store_id", "week"], how="left")
# 曜日のダミー変数を用意
df_avd_dummies=pd.get_dummies(df_avd_merged["day_of_week"])
df_avd_merged=df_avd_merged.join(df_avd_dummies)
df_ss_dummies=pd.get_dummies(df_ss_merged["day_of_week"])
df_ss_merged=df_ss_merged.join(df_ss_dummies)
df_ss_merged.head()
df_avd_merged.isna().any()
# 予約数がnaのことがあるので埋めておく
df_avd_merged=df_avd_merged.fillna(0)
df_ss_merged=df_ss_merged.fillna(0)
# 店によって推移が異なるので、air_store_idも特徴量にしておく
# int(bytes('air_00a91d42b08b08d9'.replace("air_", ""), encoding='utf-8', errors='replace').hex())
# f = lambda v: int(bytes(v.replace("air_", ""), encoding='utf-8', errors='replace').hex())%4000
# df_avd_merged["rest_id"] = df_avd_merged["air_store_id"].map(f)
# df_ss_merged["rest_id"] = df_ss_merged["air_store_id"].map(f)
# dataをair_store_idごとに分けてモデルをそれぞれ作る、で。特徴量は day_of_week, holiday_flg, reserve_visitors_x,y。
# feature_names = ["moving_ave", "holiday_flg", "reserve_visitors_x", "mean_visitors", "mean_visitors_w"] # 移動平均を入れてしまうと、ほとんど移動平均との組み合わせで木ができてしまう。testは移動平均が使えないのであまり当たらない
# feature_names = ["holiday_flg", "reserve_visitors_x", "mean_visitors", "mean_visitors_w"]

feature_names = ["Friday", "Monday", "Saturday", "Sunday","Thursday","Tuesday","Wednesday", "holiday_flg", "reserve_visitors_x", "reserve_visitors_y", "mean_visitors"]
# feature_names = ["visit_date", "Friday", "Monday", "Saturday", "Sunday","Thursday","Tuesday","Wednesday", "holiday_flg", "reserve_visitors_x", "reserve_visitors_y", "mean_visitors", "mean_visitors_w"]
traindata = df_avd_merged
testdata = df_ss_merged
# 829
store_ids= traindata.air_store_id.unique()
len(store_ids)
traindata.reserve_visitors_x.describe()
# 評価用のセル(3/4のデータを使ってfit)

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def select_data(index, df):
    return df[df["air_store_id"] == store_ids[index]]

index = 11
X = pd.DataFrame(data=select_data(index, traindata), columns=feature_names)
y = pd.DataFrame(data=select_data(index, traindata), columns=['visitors'])
threshold = len(X)//4*3

model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=1)
# model = LinearRegression()
model.fit(X[:threshold], y[:threshold])

print('Train score: {}'.format(model.score(X[:threshold], y[:threshold])))
print('Test score: {}'.format(model.score(X[threshold:], y[threshold:])))
pred = model.predict(X)

plt.plot(pred)
plt.plot(y.visitors.to_numpy())
output_list = []

for index in range(len(store_ids)):
    if index %50 == 0:
        print(index)
    train = select_data(index, traindata)
    X = pd.DataFrame(data=train, columns=feature_names)
    y = pd.DataFrame(data=train, columns=['visitors'])
    # model = LinearRegression()
    model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=1)
    model.fit(X, y)
    
    test = select_data(index, testdata)
    if len(test)==0:
        # testdataの方がtraindataよりもair_store_idが少ない
        continue
    X_test = pd.DataFrame(data=test, columns=feature_names)
    # pred=model.predict(X_test).flatten() # 線形回帰用
    pred=model.predict(X_test) # random forest用
    
    output_list.append(pd.DataFrame({'id': test.id, 'visitors': pred}))
output = pd.concat(output_list).sort_index()
output = pd.DataFrame({'id': output.id, 'visitors': output.visitors})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
output
df_ss
! head my_submission.csv
plt.plot(output_list[0].visitors)
plt.plot(output_list[26].visitors)
