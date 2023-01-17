# !pip install scipy

# !pip install plotly

# !pip install statsmodels

# !pip install scikit-learn

# !pip install matplotlib
import pandas as pd

from datetime import datetime

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import plotly.figure_factory as ff

import numpy as np

from sklearn import linear_model

import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from sklearn.preprocessing import PolynomialFeatures

import math

import os
# for dirpath, _, filenames in os.walk("/kaggle/input"):

#     for filename in filenames:

#         print(os.path.join(dirpath, filename))
DATEPARSER = lambda x: datetime.strptime(x, "%d/%m/%Y")
# get data of certain cryprocurrency by its ticker

def get_cryptcncy(ticker="BTC", start_date=None):

    # problem: how to compare a datetimeindex with a datetime

    if ticker == "BTC":

        df_crptcncy = pd.read_csv("/kaggle/input/cryptocurrency/{}.csv".format(ticker), parse_dates=["Date"],

                                  date_parser=DATEPARSER)

    else:

        df_crptcncy = pd.read_csv("/kaggle/input/cryptocurrency/{}.csv".format(ticker), parse_dates=["Date"]).iloc[::-1]

    if start_date is not None:

        start_date = datetime.strptime(start_date, "%Y-%m-%d")

        df_crptcncy = df_crptcncy.loc[df_crptcncy["Date"] >= start_date]

    df_crptcncy.set_index("Date", inplace=True)

    df_crptcncy.index.freq = "-1D"

    return df_crptcncy
# get data of external, internal, event, cryptocurrencies

def get_data(ticker="BTC", diff=None, shift=None, start_date=None):

    df_cryptcncy = get_cryptcncy("BTC", start_date=start_date)

    # external determinants

    df_gold = pd.read_csv("/kaggle/input/external/gold.csv", parse_dates=["Date"], index_col="Date")

    df_tre = pd.read_csv("/kaggle/input/external/treasury_bill.csv", parse_dates=["Date"], index_col="Date")

    df_oil = pd.read_csv("/kaggle/input/external/crude_oil.csv", parse_dates=["Date"], index_col="Date").iloc[::-1]

    df_inf = pd.read_csv("/kaggle/input/external/inflation_rate.csv", parse_dates=["Date"], index_col="Date")

    df_ue = pd.read_csv("/kaggle/input/external/USD_EUR.csv", parse_dates=["Date"], index_col="Date")

    df_uj = pd.read_csv("/kaggle/input/external/USD_JPY.csv", parse_dates=["Date"], index_col="Date")

    df_uc = pd.read_csv("/kaggle/input/external/USD_CNY.csv", parse_dates=["Date"], index_col="Date")

    # construct a df contains crypto and its factors

    if diff is None:

        df_ex = df_cryptcncy["Close"]

        columns = [ticker]

    else:

        df_ex = df_cryptcncy["Close"].diff(diff)

        columns = [ticker + "_diff {}".format(diff)]

    df_ex = pd.concat(

        [df_ex, df_gold["USD (PM)"], df_oil["value"], df_tre["1 MO"], df_ue["close"], df_uj["close"], df_uc["close"]],

        axis=1).reindex(

        df_ex.index)

    columns.extend(["Gold", "Crude Oil", "T-bill", "USD/EUR", "USD/JPY", "USD/CNY"])

    df_ex.columns = columns

    df_ex = df_ex.iloc[::-1]

    df_ex.dropna(axis=0, how="any", inplace=True)

    # df_ex.fillna(0, inplace=True)



    # internal determinants

    df_vl = pd.read_csv("/kaggle/input/internal/volume.csv", parse_dates=["Date"], index_col="Date")

    df_bs = pd.read_csv("/kaggle/input/internal/block_speed.csv", parse_dates=["Date"], index_col="Date")

    df_dff = pd.read_csv("/kaggle/input/internal/difficulty.csv", parse_dates=["Date"], index_col="Date")

    df_fee = pd.read_csv("/kaggle/input/internal/fees.csv", parse_dates=["Date"], index_col="Date")

    df_hr = pd.read_csv("/kaggle/input/internal/hash_rate.csv", parse_dates=["Date"], index_col="Date")

    df_supply = pd.read_csv("/kaggle/input/internal/supply.csv", parse_dates=["Date"], index_col="Date")

    # construct a df contains crypto and its factors

    if diff is None:

        df_in = df_cryptcncy["Close"]

        columns = [ticker]

    else:

        df_in = df_cryptcncy["Close"].diff(diff)

        columns = [ticker + "_diff {}".format(diff)]

    df_in = pd.concat(

        [df_in, df_vl["volume"], df_bs["Block Speed"], df_dff["Difficulty"], df_fee["Average"],

         df_fee["Fees Per Block"],

         df_hr["Hash Rate"], df_supply["Total Supply"]],

        axis=1).reindex(

        df_ex.index)

    columns.extend(["Volume", "Block Speed", "Difficulty", "Average", "Fees Per Block", "Hash Rate", "Total Supply"])

    df_in.columns = columns

    df_in.dropna(axis=0, how="any", inplace=True)



    # cryptocurrency makret

    columns = ["BTC", "ADA", "BCH", "DASH", "EOS", "ETH", "LTC", "IOTA", "XMR", "XRP"]

    columns.remove(ticker)

    crypto_list = []

    crypto_list.append(df_cryptcncy["Close"])

    for t in columns:

        crypto_list.append(get_cryptcncy(t)["Close"])

    df_mkt = pd.concat(crypto_list, axis=1).reindex(df_cryptcncy["Close"].index)

    columns.insert(0, ticker)

    df_mkt.columns = columns

    df_mkt.dropna(axis=0, how="any", inplace=True)

    df_mkt["CMI10"] = 0.25 * df_mkt["BTC"] + 0.25 * df_mkt["ETH"] + 0.1788 * df_mkt["XRP"] + 0.1118 * df_mkt[

        "BCH"] + 0.0667 * df_mkt["EOS"] + 0.0457 * df_mkt["LTC"] + 0.0266 * df_mkt["XMR"] + 0.0254 * df_mkt[

                          "ADA"] + 0.0220 * df_mkt["IOTA"] + 0.0229 * df_mkt["DASH"]

    df_event = pd.read_csv("/kaggle/input/events/events.csv", parse_dates=["Date"], date_parser=DATEPARSER, sep=";")

    return df_cryptcncy, df_ex, df_in, df_mkt, df_event
# correlation analysis 

def corr_cof(df, transform=None):

    if transform == "log":

        df = np.log(df)

    df_corr = df.corr(method="pearson")

    z_text = np.around(df_corr.values, decimals=3)

    fig = ff.create_annotated_heatmap(z=df_corr.values, x=list(df_corr.columns.values),

                                      y=list(df_corr.columns.values), annotation_text=z_text,

                                      showscale=True)

    fig.show()

    columns = df.columns

    rows_num = math.ceil(len(columns) / 2)

    fig = make_subplots(rows=rows_num, cols=2,

                        subplot_titles=columns)

    for index, value in enumerate(columns):

        if index == 0:

            fig.add_trace(go.Scatter(y=df[value], x=df.index, mode="markers", name=columns[0]),

                          row=1, col=1)

        else:

            fig.add_trace(go.Scatter(y=df[columns[0]], x=df[value], mode="markers", name=value),

                          row=int(index / 2 + 1), col=int(index % 2 + 1))

    fig.show()
# Multiple Linear Regression analysis

def linear_analysis(df, determinants=[], transform=None, degree=None):

    if transform == "polynomial" and degree is not None: "You should set a degree for your polynomial regression"

    X = df[determinants]

    Y = df[df.columns[0]]

    if transform == "log":

        X = np.log(X)

        Y = np.log(Y)

    elif transform == "square":

        X = np.square(X)

        Y = np.square(Y)

    elif transform == "polynomial":

        polynomial_features = PolynomialFeatures(degree=degree)

        X = polynomial_features.fit_transform(X)

    regr = linear_model.LinearRegression()

    regr.fit(X, Y)

    print('Intercept: \n', regr.intercept_)

    print('Coefficients: \n', regr.coef_)

    X = sm.add_constant(X)  # adding a constant

    model = sm.OLS(Y, X).fit()

    print(model.summary())
def differential_transform(timeseries, diff):

    timeseries_diff = timeseries.diff(periods=diff)

    timeseries_diff.fillna(0, inplace=True)

    return timeseries_diff
def unit_root_test(timeseries, method="ADF", diff=None, name=None):

    print("Name: {0}, Unit root test, Method:{1}, diff={2}".format(name, method, diff))

    if diff is not None:

        timeseries = differential_transform(timeseries, diff)

    if method == "ADF":

        timeseries_adf = adfuller(timeseries)

        print('ADF Statistic: %f' % timeseries_adf[0])

        print('p-value: %f' % timeseries_adf[1])

        print('Critical Values:')

        for key, value in timeseries_adf[4].items():

            print('\t%s: %.3f' % (key, value))
def ACF_PFC(timeseries, lags):

    fig = plt.figure()

    ax1 = fig.add_subplot(211)

    sm.graphics.tsa.plot_acf(timeseries, lags=lags, ax=ax1)

    ax2 = fig.add_subplot(212)

    sm.graphics.tsa.plot_pacf(timeseries, lags=lags, ax=ax2)

    plt.show()
def decomposing(timeseries):

    decomposition = seasonal_decompose(timeseries)

    trend = decomposition.trend

    seasonal = decomposition.seasonal

    residual = decomposition.resid

    plt.figure()

    plt.subplot(411)

    plt.plot(timeseries, label='Original')

    plt.legend(loc='best')

    plt.subplot(412)

    plt.plot(trend, label='Trend')

    plt.legend(loc='best')

    plt.subplot(413)

    plt.plot(seasonal, label='Seasonarity')

    plt.legend(loc='best')

    plt.subplot(414)

    plt.plot(residual, label='Residual')

    plt.legend(loc='best')

    plt.show()

    # problem: when to dropna and when to fillna

    trend = trend.fillna(0)

    seasonal = seasonal.fillna(0)

    residual = residual.fillna(0)

    # trend.dropna(inplace=True)

    # seasonal.dropna(inplace=True)

    # residual.dropna(inplace=True)

    return timeseries, trend, seasonal, residual
def AIC_BIC(timeseries):

    trend_evaluate = sm.tsa.arma_order_select_ic(timeseries, ic=['aic', 'bic'], trend='nc', max_ar=4,

                                                 max_ma=4)

    print('trend AIC', trend_evaluate.aic_min_order)

    print('trend BIC', trend_evaluate.bic_min_order)
def ARIMA_Model(df_close, order):

    # check stationary

    unit_root_test(df_close, diff=1)

    # ACF and PACF

    df_close_diff = differential_transform(df_close, diff=1)

    ACF_PFC(df_close_diff, lags=20)

    # decomposing

    original, trend, seasonal, residual = decomposing(df_close)

    unit_root_test(trend, diff=1, name="trend")

    unit_root_test(residual, name="residual")

    trend_diff = differential_transform(trend, diff=1)

    ACF_PFC(trend_diff, lags=20)

    ACF_PFC(residual, lags=20)

    AIC_BIC(trend_diff)

    AIC_BIC(residual)

    trend_model = ARIMA(trend, order=(1, 1, 1))

    residual_model = ARIMA(residual, (0, 0, 4))

    trend_model.fit(disp=0)

    residual_model.fit(disp=0)

    print(trend_model.summary())

    print(residual_model.summary())

    return
def draw_candlestick(df_crycency, events=None):

    fig = go.Figure()

    df_crycency = df_crycency.reset_index()

    fig.add_trace(go.Candlestick(

        x=df_crycency["Date"],

        open=df_crycency['Open'],

        high=df_crycency['High'],

        low=df_crycency['Low'],

        close=df_crycency['Close'],

        # increasing=dict(line=dict(color='#17BECF')),

        # decreasing=dict(line=dict(color='#7F7F7F'))

    ))

    for event in events.values:

        date = event[0]

        content = event[1]

        fig.add_annotation(

            x=date,

            y=df_crycency[df_crycency["Date"] == date]["Close"].values[0],

            text=content, arrowhead=3)

    fig.update_layout(height=900)

    fig.show()
ticker = "BTC"

df_cryptcncy, df_ex, df_in, df_mkt, df_event = get_data("BTC", start_date="2017-05-01")
df_btc = df_cryptcncy["Close"].iloc[::-1]
y_axis = []

x_axis = [i+1 for i in range(100)]

for lag in x_axis:

    y_axis.append(df_btc.autocorr(lag))

fig = go.Figure()

fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines"))

fig.update_layout(

    title="BTC Autocorrelation",

    xaxis_title="lag",

    yaxis_title="autocorr",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)

fig.show()
df_btc_diff = df_btc.diff(1)

y_axis = []

x_axis = [i for i in range(100)]

for lag in x_axis:

    y_axis.append(df_btc_diff.autocorr(lag+1))

fig = go.Figure()

fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines"))

fig.update_layout(

    title="BTC change Autocorrelation",

    xaxis_title="lag",

    yaxis_title="autocorr",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)

fig.show()

df_btc_delta = df_btc_diff.values[1:]/df_btc.values[:-1]

df_btc_delta = pd.Series(data=df_btc_delta)

y_axis = []

x_axis = [i for i in range(100)]

for lag in x_axis:

    y_axis.append(df_btc_delta.autocorr(lag+1))

fig = go.Figure()

fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines"))

fig.update_layout(

    title="BTC deltas Autocorrelation",

    xaxis_title="lag",

    yaxis_title="autocorr",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)

fig.show()
# external-factors analysis

df_ex_deter = ["Gold", "Crude Oil", "USD/CNY"]

corr_cof(df_ex)

linear_analysis(df_ex, df_ex_deter, "log")
# internal-factors analysis

# ["Block Speed", "Difficulty", "Average", "Fees Per Block", "Hash Rate", "Total Supply"]

df_in_deter = ["Block Speed",  "Fees Per Block", "Volume"]

corr_cof(df_in)

linear_analysis(df_in, df_in_deter)
# crypto market factors analysis

# ["BTC", "ADA", "BCH", "DASH", "EOS", "ETH", "LTC", "IOTA", "XMR", "XRP"]

corr_cof(df_mkt, "log")

df_mkt_deter = ["ETH", "XRP", "CMI10"]

linear_analysis(df_mkt, df_mkt_deter, "log")
# autocorrelation analysis analysis

unit_root_test(df_cryptcncy["Close"], name="price")

unit_root_test(df_cryptcncy["Close"], name="price_diff", diff=1)

df_diff = differential_transform(df_cryptcncy["Close"], diff=1)

ACF_PFC(df_diff, lags=50)
# event showing

draw_candlestick(df_cryptcncy, events=df_event)