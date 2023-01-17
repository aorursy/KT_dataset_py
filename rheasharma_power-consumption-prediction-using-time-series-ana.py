!pip install pmdarima



import pandas as pd

import numpy as np

from matplotlib import pyplot

from pylab import rcParams



from statsmodels.tsa.seasonal import seasonal_decompose as SDecompose

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.stattools import adfuller

from pmdarima.arima import auto_arima



pd.options.mode.chained_assignment = None



df = pd.read_csv('/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt', delimiter=';', 

                na_values=['nan','?'], dtype={'Date':str,'Time':str,'Global_active_power':np.float64,

                'Global_reactive_power':np.float64, 'Voltage':np.float64 ,'Global_intensity':np.float64,

                'Sub_metering_1':np.float64, 'Sub_metering_2':np.float64,'Sub_metering_3': np.float64})
df['DTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

df = df.drop(['Date', 'Time'], axis=1)

df = df.set_index('DTime')



df['Global_active_power'].plot(figsize=(20, 5))

df['Global_reactive_power'].plot()

pyplot.show()
print(df.isna().sum())

sampled_df = df.bfill().resample('W').mean()

print(sampled_df.corr())
sampled_df.Global_reactive_power.plot(figsize=(20, 10), color='y', legend=True)

sampled_df.Global_active_power.plot(color='r', legend=True)

sampled_df.Sub_metering_1.plot(color='b', legend=True)

sampled_df.Global_intensity.plot(color='g', legend=True)

pyplot.show()
rcParams['figure.figsize'] = 15, 6



fig, axes = pyplot.subplots(4, 2)



mul_decomposition = SDecompose(sampled_df.Global_reactive_power, model='multiplicative')

add_decomposition = SDecompose(sampled_df.Global_reactive_power, model='additive')



axes[0][0].plot(mul_decomposition.observed)

axes[0][0].set_title("Multiplicative Decomposition")

axes[1][0].plot(mul_decomposition.trend)

axes[2][0].plot(mul_decomposition.seasonal)

axes[3][0].plot(mul_decomposition.resid)



axes[0][1].plot(add_decomposition.observed)

axes[0][1].set_title("Additive Decomposition")

axes[1][1].plot(add_decomposition.trend)

axes[2][1].plot(add_decomposition.seasonal)

axes[3][1].plot(add_decomposition.resid)

pyplot.show()
split = int(0.75 * len(sampled_df))

sampled_train, sampled_test = sampled_df[:split], sampled_df[split:]



plot_acf(sampled_train.Global_reactive_power, lags=30, zero=False)

plot_pacf(sampled_train.Global_reactive_power, lags=30, zero=False)



pyplot.show()
print(adfuller(sampled_train.Global_reactive_power))



shifted_power = sampled_train.Global_reactive_power.diff(1)[1:]



print(adfuller(shifted_power))
params, seasonal_params = (1, 1, 1), (1, 0, 1, 7)



mod = SARIMAX(sampled_train.Global_reactive_power, order=params, seasonal_order=seasonal_params, 

              enforce_stationarity=False, enforce_invertibility=False)

results = mod.fit()

results.summary()
model_auto = auto_arima(sampled_train.Global_reactive_power, max_order=None, max_p=4, max_q=10, max_P=4, max_Q=10, 

                max_D=1, m=7, alpha=0.05, trend='t', information_criteria='oob', out_of_sample=int(0.02*len(sampled_train)),

                maxiter=200, suppress_warnings=True)



model_auto.summary()
model_auto = auto_arima(sampled_train.Global_reactive_power, exogenous=sampled_train[['Global_intensity', 'Sub_metering_1',  

                'Sub_metering_2', 'Sub_metering_3', 'Voltage']], max_order=None, max_p=4, max_q=10, max_P=4, max_Q=10, 

                max_D=1, m=7, alpha=0.05, trend='ct', information_criteria='oob', out_of_sample=int(0.02*len(sampled_train)),

                maxiter=200, suppress_warnings=True)



model_auto.summary()
model_auto = auto_arima(sampled_train.Global_reactive_power, exogenous=sampled_train[['Global_intensity', 'Sub_metering_1',  

                'Sub_metering_2', 'Sub_metering_3', 'Voltage']], max_order=None, max_p=4, max_q=10, max_P=4, max_Q=10, 

                max_D=1, m=7, alpha=0.05, trend=None, information_criteria='oob', out_of_sample=int(0.02*len(sampled_train)),

                maxiter=200, suppress_warnings=True)



model_auto.summary()
sampled_train['Predicted_Global_reactive_power'] = pd.DataFrame(model_auto.predict_in_sample(exogenous=

    sampled_train[['Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Voltage']]), 

    index = sampled_train.Global_reactive_power.index, columns=['Global_reactive_power'])

sampled_test['Predicted_Global_reactive_power_test'] = pd.DataFrame(model_auto.predict(n_periods=53, exogenous=

    sampled_test[['Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Voltage']]), 

     index = sampled_test.Global_reactive_power.index, columns=['Global_reactive_power'])



ax = sampled_train.Global_reactive_power.plot(figsize=(20, 5), color='red', legend=True)

sampled_test.Global_reactive_power.plot(ax=ax, color='red')



sampled_train['Predicted_Global_reactive_power'].plot(color='blue', ax=ax, legend=True)

sampled_test['Predicted_Global_reactive_power_test'].plot(color='green', ax=ax, legend=True)



pyplot.show()
resid = model_auto.resid()

resid_test = sampled_test['Predicted_Global_reactive_power_test']-sampled_test['Global_reactive_power']



ax=sampled_train['Predicted_Global_reactive_power'].plot(color='green')

sampled_test['Predicted_Global_reactive_power_test'].plot(color='green', ax=ax)



resid.plot(ax=ax, color='blue')

resid_test.plot(ax=ax, color='black')



ax.set_title('Residuals vs Predicted Values')

ax.legend(['Predicted Values', 'Residual in training set', 'Residual in testing set'])



pyplot.show()