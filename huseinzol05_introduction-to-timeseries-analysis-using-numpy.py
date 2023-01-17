!ls ../input/*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sns.set()
sales = pd.read_csv("../input/sales_train.csv")
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
sales = sales.sort_values(by='date')
sales.tail()
sales_avg_daily = sales.groupby(sales.date.dt.date).mean()
sales_avg_daily.tail()
print(sales_avg_daily.shape)
print(sales_avg_daily.iloc[0])
print(sales_avg_daily.iloc[-1])
from datetime import date
d0 = date(2013, 1, 1)
d1 = date(2015, 10, 31)
delta = d1 - d0
delta.days
def moving_average(signal, period):
    buffer = [np.nan] * period
    for i in range(period,len(signal)):
        buffer.append(signal[i-period:i].mean())
    return buffer
signal = sales_avg_daily.item_price.values
signal.shape
ma_14 = moving_average(signal, 14)
ma_21 = moving_average(signal, 21)
ma_40 = moving_average(signal, 40)
ma_90 = moving_average(signal, 90)
plt.figure(figsize=(15, 7))
plt.plot(signal, label ='sales items')
plt.plot(ma_14, label = 'ma 14 sales')
plt.plot(ma_21, label = 'ma 21 sales')
plt.plot(ma_40, label = 'ma 40 sales')
plt.plot(ma_90, label = 'ma 90 sales')
plt.legend()
plt.show()
def linear_weight_moving_average(signal, period):
    buffer = [np.nan] * period
    for i in range(period, len(signal)):
        buffer.append(
            (signal[i - period : i] * (np.arange(period) + 1)).sum()
            / (np.arange(period) + 1).sum()
        )
    return buffer
lw_ma_14 = linear_weight_moving_average(signal, 14)
lw_ma_21 = linear_weight_moving_average(signal, 21)
lw_ma_40 = linear_weight_moving_average(signal, 40)
lw_ma_90 = linear_weight_moving_average(signal, 90)
plt.figure(figsize=(15, 7))
plt.plot(signal, label ='sales items')
plt.plot(lw_ma_14, label = 'lwma 14 sales')
plt.plot(lw_ma_21, label = 'lwma 21 sales')
plt.plot(lw_ma_40, label = 'lwma 40 sales')
plt.plot(lw_ma_90, label = 'lwma 90 sales')
plt.legend()
plt.show()
def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer
anchor_3 = anchor(signal, 0.3)
anchor_5 = anchor(signal, 0.5)
anchor_8 = anchor(signal, 0.8)
plt.figure(figsize=(15, 7))
plt.plot(signal, label ='sales items')
plt.plot(anchor_3, label = 'anchor 0.3')
plt.plot(anchor_5, label = 'anchor 0.5')
plt.plot(anchor_8, label = 'anchor 0.8')
plt.legend()
plt.show()
std_signal = (signal - np.mean(signal)) / np.std(signal)
def detect(signal, treshold = 2.0):
    detected = []
    for i in range(len(signal)):
        if np.abs(signal[i]) > treshold:
            detected.append(i)
    return detected
outliers = detect(std_signal)
plt.figure(figsize=(15, 7))
plt.plot(signal)
plt.plot(signal, 'X', label='outliers',markevery=outliers, c='r')
plt.legend()
plt.show()
def removal(signal, repeat):
    copy_signal = np.copy(signal)
    for j in range(repeat):
        for i in range(3, len(signal)):
            copy_signal[i - 1] = (copy_signal[i - 2] + copy_signal[i]) / 2
    return copy_signal
def get(original_signal, removed_signal):
    buffer = []
    for i in range(len(removed_signal)):
        buffer.append(original_signal[i] - removed_signal[i])
    return np.array(buffer)
removed_signal = removal(signal, 30)
noise = get(signal, removed_signal)
plt.figure(figsize=(15, 7))
plt.subplot(2, 1, 1)
plt.plot(removed_signal)
plt.title('timeseries without noise')
plt.subplot(2, 1, 2)
plt.plot(noise)
plt.title('noise timeseries')
plt.show()
def moving_average(signal, period):
    buffer = []
    for i in range(period, len(signal)):
        buffer.append(signal[i - period : i].mean())
    return buffer
def auto_regressive(signal, p, d, q, future_count = 10):
    """
    p = the order (number of time lags)
    d = degree of differencing
    q = the order of the moving-average
    """
    buffer = np.copy(signal).tolist()
    for i in range(future_count):
        ma = moving_average(np.array(buffer[-p:]), q)
        forecast = buffer[-1]
        for n in range(0, len(ma), d):
            forecast -= buffer[-1 - n] - ma[n]
        buffer.append(forecast)
    return buffer
future_count = 120
predicted_40 = auto_regressive(signal,40,1,2,future_count)
predicted_90 = auto_regressive(signal,90,1,2,future_count)
plt.figure(figsize=(15, 7))
plt.plot(signal)
plt.plot(predicted_40, label = 'ARIMA 40 MA')
plt.plot(predicted_90, label = 'ARIMA 90 MA')
plt.legend()
plt.show()
import statsmodels.api as sm

# multiplicative
res = sm.tsa.seasonal_decompose(signal, freq=30, model="multiplicative")
fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(20, 12))
ax1.plot(res.observed)
ax1.set_ylabel('Observed')
ax2.plot(res.trend)
ax2.set_ylabel('Trend')
ax3.plot(res.seasonal)
ax3.set_ylabel('Seasonal')
ax4.plot(res.resid)
ax4.set_ylabel('Residual')
plt.show()
freq = 30
kernel = np.array([1] * freq) / freq
kernel
stride = 1
t_range = int((noise.shape[0] - freq) / stride + 1)
print('so we had to loop for:', t_range)
output_conv = np.zeros((t_range))
for i in range(t_range):
    sum_val = np.sum(signal[i * stride:i * stride + freq] * kernel)
    output_conv[i] = sum_val
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(signal)
plt.subplot(2,1,2)
plt.plot(output_conv)
plt.show()
trended = signal[:t_range] / output_conv
def seasonal_mean(x, freq):
    return np.array([np.nanmean(x[i::freq]) for i in range(freq)])
period_averages = seasonal_mean(trended, freq)
period_averages /= period_averages.mean() # normalize it with mean
period_averages
plt.plot(period_averages)
plt.show()
signal[13::13]
seasonal = np.tile(seasonal_mean(trended, freq), len(signal) // freq + 1)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(20, 12))
ax1.plot(signal)
ax1.set_ylabel('Observed')
ax2.plot(output_conv)
ax2.set_ylabel('Trend')
ax3.plot(seasonal)
ax3.set_ylabel('Seasonal')
ax4.plot(noise)
ax4.set_ylabel('Residual')
plt.show()
