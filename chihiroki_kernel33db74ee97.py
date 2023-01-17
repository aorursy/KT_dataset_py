#人に見せるつもりはなかったので、コード汚いです、すいません。
#このコードは商品コード["p1","p2","p15","p20"]についてです。
#API投げるところは板垣さんの講義のnotebookをコピペしました。
#特徴量を新たに追加していないこと、エンコもせずDataRobotに投げました。
#最後手動でアンサンブルとったかもしれません(記憶が...)
#Jupterの外で整数化しました。
# パッケージの読み込み
import pandas as pd
import numpy as np
import datetime as dt
import datarobot as dr
import lightgbm as lgb
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', 500)
num='PG1'
#FD FDW
FT_DERV_W_START=-62
FT_DERV_W_END=-1
FORECAST_W_START=1
FORECAST_W_END=62
# DataRobotのパッケージの確認
# DRの設定画面からトークンを取得
your_token = ''
# DRのAPIに接続
dr.Client(token=your_token, endpoint='https://app.datarobot.com/api/v2')
# 以下でこれまで作ったプロジェクトがリストアップされればOK
#dr.Project.list()
# データセットの読み込み
df1 = pd.read_csv('./train.csv',parse_dates=['Date'])
print(len(df1))
df1.head()
#並べ替え(気持ちの問題)
s_id=["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"]
p_id=["p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15","p16","p17","p18","p19","p20","p21","p22","p23","p24"]
df1_kari=pd.DataFrame(index=[],columns=[])
for s in s_id:
    for p in p_id:
        df1_ex=df1[(df1["store_id"] ==s) & (df1["prod_id"] == p)]
        df1_ex = df1_ex.sort_values('Date')
        #print(len(df1_kari))
        df1_kari=pd.concat([df1_kari,df1_ex],axis=0)
df1_kari['s_p']=df1_kari['store_id'] + df1_kari['prod_id']
df1_kari
#予測する店舗と商品抜き出し
#G1グループ分け(train)
s_id2=["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"]
p_id2=["p1","p2","p15","p20"]

df2_kari=pd.DataFrame(index=[],columns=[])
for s2 in s_id2:
    for p2 in p_id2:
        df2_ex=df1_kari[(df1_kari["store_id"] ==s2) & (df1_kari["prod_id"] == p2)]
        df2_ex = df2_ex.sort_values('Date')
        df2_kari=pd.concat([df2_kari,df2_ex],axis=0)

df2_kari.head()
#先にDatarobotに投げる予測値準備
#予測データの作成
import datetime
pre_kari=df2_kari.copy()
pre_kari["Date"] = pd.to_datetime(pre_kari["Date"])
pre_kari.index=pre_kari['Date']
dt1 = datetime.datetime(2014,10,31)
mae=dt1+datetime.timedelta(days=FT_DERV_W_START)
mae2=mae.strftime("%Y/%m/%d")
pre_kari1=pre_kari[mae2:]
pre_kari1.reset_index(inplace=True, drop=True)
pre_kari1
#本当に予測したい期間
# データセットの読み込み
kia_pre = pd.read_csv('./predict_kia.csv',parse_dates=['Date'])
kia_pre['s_p']=kia_pre['store_id'] + kia_pre['prod_id']
df3_kari=pd.DataFrame(index=[],columns=[])
for s2 in s_id2:
    for p2 in p_id2:
        df3_ex=kia_pre[(kia_pre["store_id"] ==s2) & (kia_pre["prod_id"] == p2)]
        df3_ex = df3_ex.sort_values('Date')
        df3_kari=pd.concat([df3_kari,df3_ex],axis=0)                                            
                                               
preG1_all=df3_kari

kia_pre_cha=preG1_all
pre_con=pd.concat([pre_kari1,kia_pre_cha],axis=0)
pre_con_final=pre_con.copy()
print(len(pre_con))
pre_con.head()
#naをffillする(train)
df_all_process = df2_kari.copy()
df_all_process[['Pct_On_Sale','Econ_ChangeGDP','EconJobsChange','AnnualizedCPI']] = df_all_process[['Pct_On_Sale','Econ_ChangeGDP','EconJobsChange','AnnualizedCPI']].fillna(method='ffill')
lag_cols = [ c for c in df_all_process.columns if c.startswith( 'lag' )]
df_all_process.dropna(axis = 0, subset = lag_cols, inplace = True)
print(df_all_process)
df_all_process.head()
# naをffillする(predict)
df_all_process_pre = pre_con.copy()
df_all_process_pre[['Pct_On_Sale','Econ_ChangeGDP','EconJobsChange','AnnualizedCPI']] = df_all_process_pre[['Pct_On_Sale','Econ_ChangeGDP','EconJobsChange','AnnualizedCPI']].fillna(method='ffill')
df_all_process_pre
#モデリングに使う特徴量セット
df_dedup = df_all_process.drop_duplicates()
df_dedup.head()
#予測データも合わせる
pre_df=df_all_process_pre.drop_duplicates()
pre_df
df_dedup2=df_dedup
df_dedup2
#いらないカラムを除去
df_dedup2=df_dedup2.drop(["store_id","prod_id"], axis=1)
pre_df=pre_df.drop(["store_id","prod_id"], axis=1)
#モデリング準備
import time
from datetime import datetime
import os
import datarobot as dr

print("DataRobot API バージョンは", dr.__version__, "を利用しています。")
print("終了:" , datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))
API_KEY=''
print("SET API KEY:" , datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))
#Datarobotへの接続
c=dr.Client(token=API_KEY,
            endpoint='https://app.datarobot.com/api/v2')

print("エンドポイントは:", c.endpoint, "TOKENは", c.token, "を利用して接続しています。")
print("接続判定:" ,c.verify)
print("終了:" , datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))
# カレンダーのアップロード
cal = dr.CalendarFile.create('calendar.csv', calendar_name='Test Calendar')
print(cal.id, cal.name)
from datarobot.enums import AUTOPILOT_MODE
from datarobot.enums import DIFFERENCING_METHOD
from datarobot.enums import TIME_UNITS
from os import path

df2=df_dedup2

#ターゲット：売上、日付項目：日付, 指標：RMSE
target='Sales_qty'
dateTimeCol = 'Date'
METRIC_NAME='MAE'

#既知の特徴量
knownFt = ['Marketing','DestinationEvent','Store_Size']
#バックテストの数
NUM_BACKTESTS = 3
VALIDATION_DURATION=None

#FD FDW
FT_DERV_W_START=FT_DERV_W_START
FT_DERV_W_END=FT_DERV_W_END
FORECAST_W_START=FORECAST_W_START
FORECAST_W_END=FORECAST_W_END

# カレンダーID
calendarId = cal.id

HOLDOUT_ENABLED=True
#HOLDOUT_START=None
HOLDOUT_START=pd.to_datetime("2014-09-01")
#HOLDOUT_DURATION=None
HOLDOUT_DURATION='P0Y0M61D'

#既知の特徴量
known_fts = [dr.FeatureSettings(fname, known_in_advance=True)
    for fname in knownFt]

#時系列パーティションの設定
partitioning_spec = dr.DatetimePartitioningSpecification(
    dateTimeCol,
    use_time_series=True,
     multiseries_id_columns=['s_p'],
    feature_derivation_window_start=FT_DERV_W_START,
    feature_derivation_window_end=FT_DERV_W_END,
    forecast_window_start=FORECAST_W_START,
    forecast_window_end=FORECAST_W_END,
    number_of_backtests=NUM_BACKTESTS,
    disable_holdout=(not HOLDOUT_ENABLED),
    holdout_start_date=HOLDOUT_START,
    holdout_duration=HOLDOUT_DURATION,
    #holdout_end_date=None,
    validation_duration=VALIDATION_DURATION,
    default_to_known_in_advance=False,
    feature_settings=known_fts,
    differencing_method=DIFFERENCING_METHOD.SEASONAL,
    periodicities=[dr.Periodicity(time_steps=7, time_unit=TIME_UNITS.DAY)],
    calendar_id=calendarId
)

NUM_WORKERS=4
#AP_MODE=AUTOPILOT_MODE.QUICK
AP_MODE=AUTOPILOT_MODE.QUICK
MAX_UPLOAD_WAIT = 30

#プロジェクト作成=データアップロード
prj_name = num
print('%s uploading...' % (prj_name))
project = dr.Project.create(df2, project_name=prj_name)
print('%s project ID: %s' % (prj_name, str(project.id)))
# オートパイロット開始
project.set_target(target,metric="MAE",
worker_count=NUM_WORKERS,
mode=AP_MODE,
partitioning_method=partitioning_spec)
project.get_models()[:10]
lb = project.get_models()
valid_models = [m for m in lb if
                m.metrics[project.metric]['crossValidation']]
best_model = min(valid_models,
                 key=lambda m: m.metrics[project.metric]['crossValidation'])
best_model
#ホールドアウトの解除
project.unlock_holdout()
train = df2
last_train_date = pd.to_datetime(train['Date']).max()

dataset = project.upload_dataset(
    pre_df,forecast_point=last_train_date
)

pred_job = best_model.request_predictions(dataset_id=dataset.id)
preds = pred_job.get_result_when_complete()


preds
#最後きれいにフォーマット
index_df_all=pre_con_final
index_df_all['Date']=pd.to_datetime(index_df_all["Date"])
index_df_all.index=index_df_all['Date']
index_df_all=index_df_all['2014-11-01':]
index_df_all.reset_index(inplace=True, drop=True)
kai=pd.DataFrame(index=[],columns=[])
kai['Date']=preds['timestamp'].str[:10]
kai['store_id']= index_df_all['store_id']
kai['prod_id']=index_df_all['prod_id']
kai['Id']=kai['Date']+'_'+kai['store_id']+'_'+kai['prod_id']
kai['Predicted']=preds['prediction']
kai_final=kai[['Id','Predicted']]
kai_final.to_csv('kai_final_'+num+'_0.csv',index=False)
kai_final
