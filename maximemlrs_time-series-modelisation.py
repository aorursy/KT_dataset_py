### Import libraries and data

import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import math

%matplotlib inline

plt.style.use('Solarize_Light2')



raw_data = pd.read_csv('../input/air-passengers/AirPassengers.csv', header=0, index_col=0, names=['data'], parse_dates=True)

raw_data.rename_axis("time", inplace=True)



# Add frequency on datetime index

freq = pd.infer_freq(raw_data.index)

raw_data = raw_data.asfreq(freq)



train = raw_data.iloc[:-12, :].copy(deep=True)

df = raw_data.copy(deep=True)



train.plot(figsize=(12,3));

plt.title("Airline passengers over time");
### Helpers

from calendar import monthrange

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.stats.diagnostic import acorr_ljungbox

from scipy.stats import shapiro



def average_per_day(df):

    """ Average the monthly quantity per day. """

    for i, row in df.itertuples():

        n_day_month = monthrange(i.year, i.month)[1]

        df.loc[i, 'data'] = row / n_day_month



        

def plot(df, col_name='data', title=None):

    """ Plot the line and its moving average smoothing and std. """

    fig, ax = plt.subplots(1,1, figsize=(12,6))

    data = ax.plot(df.index, df[col_name], label='raw data')

    mean = ax.plot(df[col_name].rolling(window=12).mean(), label="rolling mean")

    ax.set_xlabel("Time by years")

    ax.set_ylabel("Data and trend")

    

    ax_var = ax.twinx()

    var = ax_var.plot(df[col_name].rolling(window=12).std(), label="rolling std", color="#f088b3")

    ax_var.grid(False)

    ax_var.set_ylabel("Std")

    

    lines = data + mean + var

    legend = [l.get_label() for l in lines]

    ax.legend(lines, legend)

    

    plt.title(title)

    plt.tight_layout()

    fig.autofmt_xdate()



    

def plot_ACF(df, col_name='data', n_lags=20):

    """ Plot ACF/PACF graphs. """

    fig, ax = plt.subplots(2, figsize=(12,6))

    ax[0] = plot_acf(df[col_name].dropna(), ax=ax[0], lags=n_lags)

    ax[1] = plot_pacf(df[col_name].dropna(), ax=ax[1], lags=n_lags)

    



def differentiate(df, k=1, col_data='data'):

    "Differentiate the time series by substracting values at instant t-1...t-k"

    values = [np.NaN for i in range(k)]

    for t in range(k, len(df)):

        val = df[col_data][t] - df[col_data][t-k]

        values.append(val)



    return pd.DataFrame(values, index=df.index, columns=[col_data])



    

def test_ADF(df, col_name='data'):

    """ Test unilatéral droit : Plus la stat de test est négative, plus H0 est rejeté. """

    print("> ADF test : H0 = there is a unit root. H1 = The process has a root outside the unit circle (=stationarity)")

    message = ['[H0 accepted] There is a unit root', '[H0 rejected] The process is stationary']

    dftest = adfuller(df[col_name].dropna(), autolag='AIC')

    print("Test statistic = {:.3f}".format(dftest[0]))

    print("P-value = {:.3f}".format(dftest[1]))

    print("Critical values :")

    for k, v in dftest[4].items():

        result_test = 1 - int(dftest[0] > v)  # 0 if H0 accepted

        print("\t{}: {} - {} with {}% confidence".format(k, v, message[result_test], 100-int(k[:-1])))





def test_KPSS(df, col_name='data', has_trend=False):

    print("> KPSS test : H0 = the data is stationary. H1 = There is a unit root.")

    message = ['[H0 accepted ] The data is stationary', '[H0 rejected] There is a unit root']

    regression = 'ct' if has_trend else 'c'

    dftest = kpss(df[col_name].dropna(), regression=regression)

    print("Test statistic = {:.3f}".format(dftest[0]))

    print("P-value = {:.3f}".format(dftest[1]))

    print("Critical values :")

    for k, v in dftest[3].items():

        result_test = 1 - int(dftest[0] < v)  # 0 if H0 accepted

        print("\t{}: {} - {} with {}% confidence".format(k, v, message[result_test], 100-float(k[:-1])))

        



def test_LjungBox(res, lags=None, has_seasonality=False, seasonal_period=0, alpha=0.05):

    print("\n> Ljung-Box test : H0 : no autocorrelation between lags 1 to r(=independent distrib). H1 : autocorrelation between lags 1 to r.")

    # Compute the lags

    if has_seasonality is False and lags is None:

        lags = min(10, int(len(res)/5))

    elif lags is None:

        lags = min((len(res) // 2 - 2), 40)  # Default value according to the doc.

    

    message = ["[H0 accepted] No autocorrelation between lags 1 to %s" % lags, "[H0 rejected] Autocorrelation between lags 1 to %s" % lags]

    

    ljbtest = acorr_ljungbox(res, lags=lags)

    p_val = ljbtest[1][len(ljbtest[1]) - 1]

    

    print("P-value = {:.3f}".format(p_val))

    print(message[alpha > p_val])

    

    

def test_shapiro(sample, alpha=0.05):

    print("\n> Shapiro-Wilk test : H0 = the data is drawn from normal distribution.")

    message = ['[H0 accepted] The data is drawn from normal distribution', '[H0 rejected] The data isn\'t from normal distribution.']

    p_val = shapiro(sample)[1]

    print("P-value : {:.3f}".format(p_val))

    print(message[alpha > p_val])

        

def rmse(x1, x2):

    " Compute the RMSE given two numpy array or pandas Series. "

    assert len(x1) == len(x2), 'The two samples must be of same size'

    # If the sample given is a pd.Series, get the array of values instead

    if isinstance(x1, pd.Series):

        x1 = x1.values

    if isinstance(x2, pd.Series):

        x2 = x2.values

        

    T = len(x1)

    rmse = math.sqrt(np.sum((x1 - x2)**2)/ T)

    return rmse



def mape(forecast, data):

    " Compute the MAPE = 1/T * SUM(|data[i] - forecast[i]| / data[i]). Metric returned in percentage. "

    assert len(forecast) == len(data), 'The two samples must be of same size'

    if isinstance(forecast, pd.Series):

        forecast = forecast.values

    if isinstance(data, pd.Series):

        data = data.values

        

    T = len(forecast)

    mape = 100 * np.sum(abs(forecast - data) / data) / T

    return round(mape, 3)
# Display the rolling mean and standard deviation

plot(train, title="Airline passengers over time")
### Holt Winter exponential smoothing

from statsmodels.tsa.holtwinters import ExponentialSmoothing



# We'll chose an additive trend and multiplicative seasonality because the size of the latter pattern depends on the level of the TS.

hw_model = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12).fit()

fitted_val = hw_model.fittedvalues

prediction = hw_model.predict(df.index[len(train)], df.index[-1])
from statsmodels.graphics.gofplots import qqplot

from scipy.stats import norm

from statsmodels.nonparametric.kde import KDEUnivariate





# Plot the data along with the fitted values and the predictions of the model

fig, ax = plt.subplots(1,1, figsize=(12,6))

ax.plot(df.index, df['data'], label='Data')

ax.plot(df.index[:len(train)-1], fitted_val[:len(train)-1], label="Model on train data (AICC={:.3f})".format(hw_model.aicc), color="#f088b3")

ax.plot(df.index[len(train):], prediction, label="Forecast")

ax.set_xlabel("Time by years")

ax.set_ylabel("Monthly number of passengers")

ax.legend()

plt.title("Holt Winter exponential smoothing model")

plt.tight_layout()

fig.autofmt_xdate()



# Evaluate the model with RMSE

train_RMSE_hw = rmse(fitted_val[:len(train)-1], df['data'][:len(train)-1])

train_MAPE_hw = mape(fitted_val[:len(train)-1], df['data'][:len(train)-1])

test_RMSE_hw = rmse(prediction, df['data'][len(train):])

test_MAPE_hw = mape(prediction, df['data'][len(train):])

print("\nTrain RMSE : {:.3f}\nTest RMSE : {:.3f}\nTrain MAPE : {:.3f} %\nTest MAPE : {:.3f} %".format(train_RMSE_hw, test_RMSE_hw, train_MAPE_hw, test_MAPE_hw))



# Analyse residuals : plot the standardized residuals (divided by its std), a correlogram, a QQ-plot and an histogram

residuals = (df['data'].values - pd.concat([fitted_val, prediction]))

standardized_residuals = residuals / residuals.std()



fig, ax = plt.subplots(2,2, figsize=(12,12))

# Plot of the standardized residuals

ax[0, 0].plot(df.index, standardized_residuals)

ax[0, 0].axhline(standardized_residuals.mean(),color='r',ls='--', label='Mean')

ax[0, 0].set_title("Standardized Residuals")

ax[0, 0].legend()



# ACF of the residuals

ax[0, 1] = plot_acf(standardized_residuals, ax=ax[0, 1], lags=20)



#QQplot

ax[1, 0] = qqplot(standardized_residuals, ax=ax[1, 0], line='45')



# Histogram and kernel density estimation

ax[1, 1].hist(standardized_residuals, density=True, bins=20)

normal_distrib = norm.pdf(np.linspace(-3,4), 0, 1)

ax[1,1].plot(np.linspace(-3,4), normal_distrib, label="N(0,1)", color='r')



kde = KDEUnivariate(standardized_residuals)

kde.fit(clip=(-3, 4))

ax[1,1].plot(kde.support, kde.density, label="KDE", color='g')

ax[1,1].legend(loc="upper right")

ax[1, 1].set_title("Histogram of standardized residuals")



# Test if there is correlation between the residuals

test_LjungBox(residuals, lags=20, has_seasonality=True, seasonal_period=12)



# Test if the residuals is drawn from a normal distribution

test_shapiro(residuals)
# Log transformation to have same magnitude of variance over time

log_train = train.data.apply(lambda x: math.log(x)).to_frame()



plot(log_train, title="Log of airline passengers over time")
# Removing the trend and seasonality pattern by differing

stationary_data = pd.DataFrame(differentiate(log_train.copy(deep=True), 12), columns=['data'])

stationary_data = differentiate(stationary_data, 1)
# Checking the stationarity

plot(stationary_data, title='Stationary data')

plot_ACF(stationary_data, n_lags=36)

test_ADF(stationary_data)

test_KPSS(stationary_data)
# Search grid to choose the best SARIMA parameters using the software X-13ARIMA-SEATS made by the Census Bureau

# It is needed to install the software (winx13 or winx12) at https://www.census.gov/srd/www/winx13/winx13_down.html



from statsmodels.tsa.x13 import x13_arima_select_order



path_to_folder = 'your_path_to_the_downloaded_folder'

path_winx13 = os.path.join(path_to_folder, 'x13as', 'x13as')



# I don't know how to import winx13 on Kaggle kernel so it's not working.



#x13_order = x13_arima_select_order(log_train, x12path=path_winx13)



#print("ARIMA parameter order : %s\nSeasonal ARIMA parameter order : %s" % (x13_order.order, x13_order.sorder))
# ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX



# Explanation below for the substraction of the first value to the data

first_val = log_train['data'][0]



sarima_model = SARIMAX(log_train['data'] - first_val, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))

sarima_fit = sarima_model.fit()



print(sarima_fit.summary())
test = df[-12:]



sarima_pred = sarima_fit.get_prediction('1960-01', '1960-12')

predicted_means = sarima_pred.predicted_mean.apply(lambda x: math.exp(x + first_val))

predicted_intervals = sarima_pred.conf_int(alpha=0.05)

lower_bounds = predicted_intervals['lower data'].apply(lambda x: math.exp(x + first_val))

upper_bounds = predicted_intervals['upper data'].apply(lambda x: math.exp(x + first_val))



train_RMSE_sarima = rmse(sarima_fit.fittedvalues.apply(lambda x: math.exp(x + first_val))[1:], raw_data['data'][1:-12])

test_RMSE_sarima = rmse(test['data'], predicted_means)

train_MAPE_sarima = mape(sarima_fit.fittedvalues.apply(lambda x: math.exp(x + first_val))[1:], raw_data['data'][1:-12])

test_MAPE_sarima = mape(test['data'], predicted_means)



fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(df.data.index[:-12], df.data.values[:-12]);

ax.plot(test.index, test.values, label='truth');

ax.plot(test.index, predicted_means, color='#ff7823', linestyle='--', label="prediction (RMSE={:0.2f})".format(test_RMSE_sarima));

ax.plot(df.data.index[:-12], sarima_fit.fittedvalues.apply(lambda x: math.exp(x + first_val)), label="SARIMA model")

ax.fill_between(test.index, lower_bounds, upper_bounds, color='#ff7823', alpha=0.3, label="confidence interval (95%)");

ax.legend();

ax.set_title("SARIMA");
# Residual diagnostic

print(sarima_fit.plot_diagnostics(figsize=(12,12)))
residuals = pd.DataFrame(sarima_fit._results.forecasts_error[0], columns=['data'])

test_LjungBox(residuals, has_seasonality=True, seasonal_period=12)

test_shapiro(standardized_residuals)
### Comparing the models



print("# Modèle Holt-Winter\nTrain RMSE : {:.3f}\nTest RMSE : {:.3f}\nTrain MAPE : {:.3f} %\nTest MAPE : {:.3f} %\n\n".format(train_RMSE_hw, test_RMSE_hw, train_MAPE_hw, test_MAPE_hw))



print("# Modèle SARIMA\nTrain RMSE : {:.3f}\nTest RMSE : {:.3f}\nTrain MAPE : {:.3f} %\nTest MAPE : {:.3f} %\n\n".format(train_RMSE_sarima, test_RMSE_sarima, train_MAPE_sarima, test_MAPE_sarima))