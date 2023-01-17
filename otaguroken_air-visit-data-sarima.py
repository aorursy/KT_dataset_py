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
! head hpg_store_info.csv -n 2
! head sample_submission.csv -n 2
! head air_visit_data.csv -n 2
! head date_info.csv -n 2
! head air_reserve.csv -n 2
! head hpg_reserve.csv -n 2
! head air_store_info.csv -n 2
! head store_id_relation.csv -n 2
import pandas as pd
df = pd.read_csv(files[2])
# import pandas_profiling as pdp
# pdp.ProfileReport(df)
! cat air_visit_data.csv | grep air_ba937bf13d40fb24 | wc -l
df.groupby("air_store_id").count()
from matplotlib import pylab as plt
plt.plot(df.query("air_store_id=='air_034a3d5b40d5b1b1'").visitors)
plt.plot(df.query("air_store_id=='air_0241aa3964b7f861'").visitors)
import statsmodels.api as sm
#自己相関係数 => 7で周期がありそう
sm.graphics.tsa.plot_acf(df.query("air_store_id=='air_0241aa3964b7f861'").visitors, lags=40)
plt.xlabel('lags')
plt.ylabel('corr')
sm.graphics.tsa.plot_acf(df.query("air_store_id=='air_034a3d5b40d5b1b1'").visitors, lags=40)
plt.xlabel('lags')
plt.ylabel('corr')
# そもそも欠損のないデータになっているのか確認
pd.set_option('display.max_rows', 400)
print(df.query("air_store_id=='air_0241aa3964b7f861'").visit_date)
# 229752    2016-09-11
# 229753    2016-09-13 欠損している
# 欠損を埋めるためにstore_idごとにデータを保持
df_dict = {}
for name, group in df.groupby('air_store_id'):
    df_dict[name] = group
df_dict['air_00a91d42b08b08d9']
df_dict['air_00a91d42b08b08d9']['date'] = pd.to_datetime(df_dict['air_00a91d42b08b08d9']['visit_date'], format='%Y-%m-%d')
df_dict['air_00a91d42b08b08d9'].index = df_dict['air_00a91d42b08b08d9'].date
df_dict['air_00a91d42b08b08d9'].resample('D').median()
visitors = df_dict['air_00a91d42b08b08d9'].resample('D').median().fillna('0')
import math
raw_value = [_ for _ in visitors.visitors]
raw_value
sm.graphics.tsa.plot_acf(raw_value, lags=40)
plt.xlabel('lags')
plt.ylabel('corr')
# 欠損を埋める前。だいぶ良くなっている。(欠損に周期性があるが。。)
sm.graphics.tsa.plot_acf(df.query("air_store_id=='air_00a91d42b08b08d9'").visitors, lags=40)
plt.xlabel('lags')
plt.ylabel('corr')
# 偏自己相関も調べる (https://qiita.com/savaniased/items/01eb7de33495b2efaad2 がわかりやすい)
sm.graphics.tsa.plot_pacf(raw_value, lags=40)
plt.xlabel('lags')
plt.ylabel('corr')
# 欠損を埋めるためにstore_idごとにデータを保持
df_dict = {}
for name, group in df.groupby('air_store_id'):
    df_dict[name] = group
# 欠損を0埋め
for key in df_dict:
    dataframe = df_dict[key]
    dataframe['date'] = pd.to_datetime(dataframe['visit_date'], format='%Y-%m-%d')
    dataframe.index = dataframe.date
    # resampleは日付でgroupbyする (欠損行はnanで埋められる)
    dataframe = dataframe.resample('D').median().fillna(0)
    dataframe['air_store_id'] = key
    df_dict[key] = dataframe
df_dict['air_00a91d42b08b08d9']
df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy()
# 差分を見たい (arimaのdを決めたい, 1回だけ差を取るのはdが1) arモデルが自己回帰, maモデルが移動平均, iが差。トレンドを除くために差を取って、arモデルとmaモデルを使う
ts = df_dict['air_00a91d42b08b08d9']['visitors']
diff = ts - ts.shift()
diff.head()
plt.plot(diff)
# 7で周期があるので、7周期で。sarimaの周期性分のsは7にする
sm.graphics.tsa.plot_acf(diff.to_numpy()[1:], lags=40)
plt.xlabel('lags')
plt.ylabel('corr')
# diffで100人増加しているところがあるのでその影響で、変なピークがありそう <= 予約情報とか重要かもしれない
sm.graphics.tsa.plot_pacf(diff.to_numpy()[1:], lags=50)
plt.xlabel('lags')
plt.ylabel('corr')
# 適当な値を入れてみる
sarimax = sm.tsa.SARIMAX(df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy(), 
                        order=(3, 3, 3),
                        seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity = False,
                        enforce_invertibility = False
                        ).fit()
sarimax.summary()
# res = sm.tsa.arma_order_select_ic(df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy(), max_ar=7, max_ma=7, ic='aic')
# 'aic_min_order': (6, 4)
# res
# よって 直前6日で自己回帰、4日の移動平均が良い。(ARMAの場合は) IとSがあるので

# seasonal_order の1のところは0~2で探索した方が良いらしい
# (6, 1, 4), (1, 1, 1, 28)でAICは1889.643
# (6, 1, 4), (1, 1, 1, 7)でAICは2174.805
# (6, 0, 4), (1, 1, 1, 7)でAICは2173.757
# (4, 1, 4), (1, 1, 3, 7)でAICは2078.872
# (4, 1, 4), (2, 1, 3, 7)でAICは2080.431
# (6, 1, 4), (1, 0, 1, 14)でAICは2182.977
# (6, 1, 4), (2, 1, 3, 7)で 2084.391
# (6, 1, 4), (4, 1, 2, 7)で 2020.632 <=これが良さそう
# (6, 1, 4), (1, 1, 1, 14)でAICは2088.864
# (6, 1, 4), (1, 1, 1, 35)でAICは1781.245 (重い)
# (6, 1, 4), (1, 1, 1, 42)でAICは1679.772 (重い) 増やすほど増えるので過学習しているかも 。42にすればパラメータ数も42 * 2くらいに。。
# (4, 1, 2), (1, 1, 1, 7)でAICは2183.734
# (4, 1, 2), (4, 1, 2, 7)でAICは2033.271
# (5, 1, 3), (4, 1, 2, 7)でAICは2033.847
sarimax = sm.tsa.SARIMAX(df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy(), 
                        order=(6, 1, 4),
                        seasonal_order=(4, 1, 2, 7),
                        enforce_stationarity = False,
                        enforce_invertibility = False
                        ).fit()
sarimax.summary()
train_pred = sarimax.predict()
from sklearn.metrics import mean_squared_error
train_rmse = np.sqrt(mean_squared_error(df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy(), train_pred))
train_rmse
import matplotlib.pyplot
plt.plot(train_pred)
plt.plot(df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy())
len(df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy())
df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy().sum()
6051 / 296
np.sqrt(mean_squared_error(df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy(), [20.44256 for _ in range(296)]))
# 0があたっているというだけであまり良くない結果のように思う。。
import matplotlib.pyplot
plt.plot(sarimax.predict(start=0, end = 400))
plt.plot(df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy())
plt.plot(sarimax.predict(start=0, end = 400))
np.sqrt(mean_squared_error(df_dict['air_00a91d42b08b08d9']['visitors'].to_numpy(), sarimax.predict()))
! cat sample_submission.csv | grep air_00a91d42b08b08d9 | head -n 1
! cat sample_submission.csv | grep air_00a91d42b08b08d9 | tail -n 1
! cat air_visit_data.csv | grep air_00a91d42b08b08d9 | head -n 1
! cat air_visit_data.csv | grep air_00a91d42b08b08d9 | tail -n 1
! cat sample_submission.csv | grep air_ffcc2d5087e1b476 | head -n 1
! cat sample_submission.csv | grep air_ffcc2d5087e1b476 | tail -n 1
! cat air_visit_data.csv | grep air_ffcc2d5087e1b476 | head -n 1
! cat air_visit_data.csv | grep air_ffcc2d5087e1b476 | tail -n 1
! cat sample_submission.csv | grep air_fee8dcf4d619598e | head -n 1
! cat sample_submission.csv | grep air_fee8dcf4d619598e | tail -n 1
! cat air_visit_data.csv | grep air_fee8dcf4d619598e | head -n 1
! cat air_visit_data.csv | grep air_fee8dcf4d619598e | tail -n 1
# web上のSARIMAのわかりやすい説明。
# https://www.ai.u-hyogo.ac.jp/~arima/arima.pdf
# 移動平均と自己回帰と1週間前の値を今回の場合は、特徴量に入れるだけでも良いかも。