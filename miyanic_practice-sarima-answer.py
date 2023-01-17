import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm 

df = pd.read_csv('/kaggle/input/insurance/test_Insurance.csv')
Insurance = pd.Series(df['#Insurance'], dtype='float') 
Insurance.index = pd.to_datetime(df['Month']) 
Insurance.plot()
res = sm.tsa.seasonal_decompose(Insurance) # 解析結果は取得

original = Insurance # オリジナルデータ
trend = res.trend # トレンドデータ
seasonal = res.seasonal # 季節性データ
residual = res.resid # 残差データ
plt.figure(figsize=(8, 8))

# オリジナルデータのプロット
plt.subplot(4,1,1)
plt.plot(original)
plt.ylabel('Original')

# トレンド性のプロット
plt.subplot(4,1,2)
plt.plot(trend)
plt.ylabel('Trend')

# 季節性のプロット
plt.subplot(4,1,3)
plt.plot(seasonal)
plt.ylabel('Seasonality')

# 残差のプロット
plt.subplot(4,1,4) 
plt.plot(residual)
plt.ylabel('Residuals')


# ラグ0~30までの自己相関係数の出力
Insurance_acf = sm.tsa.stattools.acf(Insurance, nlags=30)

# ラグ0~30までの偏自己相関係数の出力
Insurance_pacf = sm.tsa.stattools.pacf(Insurance, nlags=30)

fig = plt.figure(figsize=(8, 8))

# ラグ0~30までの自己相関係数のプロット(コレログラム)
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(Insurance, lags=30, ax=ax1) 

# ラグ0~30までの偏自己相関係数のプロット(コレログラム)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(Insurance, lags=30, ax=ax2)

plt.tight_layout()

# SARIMAパラメータ最適化（総当たりチェック）
import warnings
warnings.filterwarnings('ignore')

#時系列ARIMA次数設定(AR:1~3,階差0~1,MA:0~3)
min_p = 1; max_p = 3
min_d = 0; max_d = 1
min_q = 0; max_q = 3 

#周期性ARIMA次数設定(AR:0~1,階差0~1,MA:0~1)
min_sp = 0; max_sp = 1
min_sd = 0; max_sd = 1
min_sq = 0; max_sq = 1

test_pattern = (max_p - min_p +1)*(max_q - min_q + 1)*(max_d - min_d + 1)*(max_sp - min_sp + 1)*(max_sq - min_sq + 1)*(max_sd - min_sd + 1)
print("pattern:", test_pattern)

#周期設定
sfq = 12

#学習データ設定
ts = Insurance

#総当たりで検証
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

#AICが最小の最適モデル出力                        
print("best[aic] parameter ********")
print(test_results[test_results.aic == min(test_results.aic)])


Insurance_train = Insurance['1984-04':'2019-03'] # 学習データ（1984/04~2019/03）
Insurance_test = Insurance['2019-04':'2020-02'] # 検証データ（2019/04~2020/02）

# 学習データからSARIMAモデル（周期12,時系列ARIMA(AR:2,階差:1,MA:3),周期性ARIMA(AR:1,階差:1,MA:1)）作成
sarimax_train = sm.tsa.SARIMAX(Insurance_train, 
                        order=(2, 1, 3),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity = False,
                        enforce_invertibility = False
                        ).fit()

# 作成したSARIMAモデルから2019/04~2020/02を予測(検証データの範囲)
sarimax_train_pred = sarimax_train.predict('2019-04','2020-02')
plt.figure(figsize=(8, 4))

# 実データと予測結果をプロット
plt.plot(Insurance, label="actual")
plt.plot(sarimax_train_pred, c="r", label="model-pred")
plt.legend(loc='best')

# ARIMAパラメータ最適化（総当たりチェック）
import warnings
warnings.filterwarnings('ignore')

#ARIMA次数設定(AR:1~3,階差0~1,MA:0~3)
min_p = 1; max_p = 3
min_d = 0; max_d = 1
min_q = 0; max_q = 3 

test_pattern = (max_p - min_p +1)*(max_q - min_q + 1)*(max_d - min_d + 1)
print("pattern:", test_pattern)

#学習データ設定
ts = Insurance

#総当たりで検証
test_results = pd.DataFrame(index=range(test_pattern), columns=["model_parameters", "aic"])
num = 0
for p in range(min_p, max_p + 1):
    for d in range(min_d, max_d + 1):
        for q in range(min_q, max_q + 1):
                        arima =sm.tsa. ARIMA(ts, order=(p,d,q)).fit()
                        test_results.iloc[num]["model_parameters"] = "order=(" + str(p) + ","+ str(d) + ","+ str(q) + ")"
                        test_results.iloc[num]["aic"] = arima.aic
                        print(num,'/', test_pattern-1, test_results.iloc[num]["model_parameters"],  test_results.iloc[num]["aic"] )
                        num = num + 1

#AICが最小の最適モデル出力  
print("best[aic] parameter ********")
print(test_results[test_results.aic == min(test_results.aic)])
Insurance_train = Insurance['1984-04':'2019-03'] # 学習データ（1984/04~2019/03）
Insurance_test = Insurance['2019-04':'2020-02'] # 検証データ（2019/04~2020/02）

# 学習データからARIMAモデル(AR:2,階差:1,MA:3)作成
arima_model =sm.tsa. ARIMA(Insurance_train, order=(2,1,3)).fit()
arima_model_pred = arima_model.predict('2019-04','2020-02')

plt.figure(figsize=(8, 4))

# 実データと予測結果をプロット
plt.plot(Insurance, label="actual")
plt.plot(arima_model_pred, c="r", label="model-pred")
plt.legend(loc='best')
from sklearn.metrics import mean_squared_error

print("最適SARIMAモデルRMSE")
print(np.sqrt(mean_squared_error(Insurance_test, sarimax_train_pred)))

print("最適ARIMAモデルRMSE")
print(np.sqrt(mean_squared_error(Insurance_test, arima_model_pred)))
