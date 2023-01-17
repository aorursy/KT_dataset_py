import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm 


df = pd.read_csv('/kaggle/input/airpassengers/AirPassengers.csv')
passengers = pd.Series(df['#Passengers'], dtype='float') 
passengers.index = pd.to_datetime(df['Month']) 
passengers.plot()

res = sm.tsa.seasonal_decompose(passengers) # 解析結果は取得

original = passengers # オリジナルデータ
trend = res.trend # トレンドデータ
seasonal = res.seasonal # 季節性データ
residual = res.resid # 残差データ

plt.figure(figsize=(8, 8)) # グラフ描画枠作成、サイズ指定

# オリジナルデータのプロット
plt.subplot(4,1,1) # グラフ4行1列の1番目の位置（一番上）
plt.plot(original)
plt.ylabel('Original')

# trend データのプロット
plt.subplot(4,1,2) # グラフ4行1列の2番目の位置
plt.plot(trend)
plt.ylabel('Trend')

# seasonalデータ のプロット
plt.subplot(4,1,3) # グラフ4行1列の3番目の位置
plt.plot(seasonal)
plt.ylabel('Seasonality')

# residual データのプロット
plt.subplot(4,1,4) # グラフ4行1列の4番目の位置（一番下）
plt.plot(residual)
plt.ylabel('Residuals')
# 自己相関係数の出力
passengers_acf = sm.tsa.stattools.acf(passengers, nlags=30) #ラグ=30

# 偏自己相関係数の出力
passengers_pacf = sm.tsa.stattools.pacf(passengers, nlags=30) #ラグ=30


fig = plt.figure(figsize=(8, 8))

# 自己相関(ACF)のグラフ
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(passengers, lags=30, ax=ax1) #ACF計算とグラフ自動作成

# 偏自己相関(PACF)のグラフ
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(passengers, lags=30, ax=ax2) #PACF計算とグラフ自動作成

plt.tight_layout() # グラフ間スキマ調整 

# SARIMAパラメター最適化（総当たりチェック）
import warnings
warnings.filterwarnings('ignore') # 警告非表示（収束：ConvergenceWarning）

# パラメータ範囲
# order(p, d, q)
min_p = 1; max_p = 3 # min_pは1以上を指定しないとエラー
min_d = 0; max_d = 1
min_q = 0; max_q = 3 

# seasonal_order(sp, sd, sq)
min_sp = 0; max_sp = 1
min_sd = 0; max_sd = 1
min_sq = 0; max_sq = 1

test_pattern = (max_p - min_p +1)*(max_q - min_q + 1)*(max_d - min_d + 1)*(max_sp - min_sp + 1)*(max_sq - min_sq + 1)*(max_sd - min_sd + 1)
print("pattern:", test_pattern)

sfq = 12 # seasonal_order周期パラメータ
ts = passengers # 時系列データ

test_results = pd.DataFrame(index=range(test_pattern), columns=["model_parameters", "aic"])
num = 0
for p in range(min_p, max_p + 1):
    for d in range(min_d, max_d + 1):
        for q in range(min_q, max_q + 1):
            for sp in range(min_sp, max_sp + 1):
                for sd in range(min_sd, max_sd + 1):
                    for sq in range(min_sq, max_sq + 1):
                        sarima = sm.tsa.SARIMAX(
                            ts, order=(p, d, q), 
                            seasonal_order=(sp, sd, sq, sfq), 
                            enforce_stationarity = False, 
                            enforce_invertibility = False
                        ).fit()
                        test_results.iloc[num]["model_parameters"] = "order=(" + str(p) + ","+ str(d) + ","+ str(q) + "), seasonal_order=("+ str(sp) + ","+ str(sd) + "," + str(sq) + ")"
                        test_results.iloc[num]["aic"] = sarima.aic
                        print(num,'/', test_pattern-1, test_results.iloc[num]["model_parameters"],  test_results.iloc[num]["aic"] )
                        num = num + 1

# 結果（最小AiC）
print("best[aic] parameter ********")
print(test_results[test_results.aic == min(test_results.aic)])

# オリジナル passengers 1949-01 ～ 1960-12
passengers_train2 = passengers['1949-01':'1959-12'] # モデル作成用データ（訓練）1年テスト用残し
passengers_test2 = passengers['1960-01':'1960-12'] # テスト用データ1年分

# SRIMAモデル（テストデータ1年を除いてモデル作成）
sarimax_train = sm.tsa.SARIMAX(passengers_train2, 
                        order=(3, 1, 3),
                        seasonal_order=(0, 1, 1, 12),
                        enforce_stationarity = False,
                        enforce_invertibility = False
                        ).fit()

sarimax_train2_pred = sarimax_train.predict('1960-1', '1960-12') # テストデータ1年分予測
plt.figure(figsize=(8, 4))

plt.plot(passengers, label="actual")
plt.plot(sarimax_train2_pred, c="r", label="model-pred")
plt.legend(loc='best')
from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(passengers_test2, sarimax_train2_pred)))
# オリジナル passengers 1949-01 ～ 1960-12
passengers_train2 = passengers['1949-01':'1959-12'] # モデル作成用データ（訓練）1年テスト用残し
passengers_test2 = passengers['1960-01':'1960-12'] # テスト用データ1年分

# ARIMAモデル（テストデータ1年を除いてモデル作成）
arima_model =sm.tsa. ARIMA(passengers_train2, order=(3,1,3)).fit()
arima_mode_pred = arima_model.predict('1960-1', '1960-12') # テストデータ1年分予測
plt.plot(passengers, label="actual")
plt.plot(arima_mode_pred, c="r", label="arima-model-pred")
plt.legend(loc='best')

print(np.sqrt(mean_squared_error(passengers_test2, arima_mode_pred)))
