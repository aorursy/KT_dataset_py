import pandas as pd

import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import graph_objs as go

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import warnings

init_notebook_mode(connected = True)
# Импортируем данные с курсом доллара и преобразуем их в нужный формат

usd = pd.read_csv("../input/RC_F01_01_2011_T07_12_2019.csv", sep=';', decimal=',')

usd.head(3)

usd.data = pd.to_datetime(usd.data)
# Взглянем на данные

usd.head()
# и их краткие характеристики

usd.info()
# Перегруппируем ряд из ежедневной в ежемесячную динамику

grouped = usd.groupby('data').mean().resample('w').ffill()
# Функция графиков для отрисовки динамики курса

def plotly_df(df, title = ''):

    data = []



    for column in df.columns:

        trace = go.Scatter(

            x = df.index,

            y = df[column],

            mode = 'lines',

            name = column

        )

        data.append(trace)



    layout = dict(title = title)

    fig = dict(data = data, layout = layout)

    iplot(fig, show_link=False)
plotly_df(grouped, title = "Динамика курса доллара c 2011-го года")
data = grouped['2016-03-01':]
plotly_df(data, title = "Динамика курса доллара с 2016-го года")
# Функция графиков для оценки стационарности ряда

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

    with plt.style.context(style):    

        fig = plt.figure(figsize=figsize)

        layout = (2, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1, 0))

        pacf_ax = plt.subplot2grid(layout, (1, 1))



        y.plot(ax=ts_ax)

        ts_ax.set_title('Time Series')

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)



        print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(y)[1])



        plt.tight_layout()

    return
tsplot(data.curs, lags=30)

warnings.filterwarnings("ignore")
data['curs_diff'] = data['curs'].diff(periods=1)

tsplot(data['curs_diff'].dropna(axis=0), lags=30)
ps = range(0, 5)

d=1

qs = range(0, 4)

Ps = range(0, 5)

D=1

Qs = range(0, 1)



from itertools import product



parameters = product(ps, qs, Ps, Qs)

parameters_list = list(parameters)

print('Необходимо "перебрать"', len(parameters_list), 'параметров, подходящих для модели SARIMA')
%%time

results = []

best_aic = float("inf")



for param in parameters_list:

    #try except нужен, потому что на некоторых наборах параметров модель не обучается

    try:

        # наблюдаемый процесс (ряд - серия), внешние регрессоры

        model=sm.tsa.statespace.SARIMAX(data['curs_diff'].dropna(axis=0), order=(param[0], d, param[1]), 

                                        seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)

    #выводим параметры, на которых модель не обучается и переходим к следующему набору

    except ValueError:

        print('wrong parameters:', param)

        continue

    aic = model.aic

    #сохраняем лучшую модель, aic, параметры

    if aic < best_aic:

        best_model = model

        best_aic = aic

        best_param = param

    results.append([param, model.aic])



warnings.filterwarnings("ignore")



result_table = pd.DataFrame(results)

result_table.columns = ['parameters', 'aic']

print(result_table.sort_values(by = 'aic', ascending=True).head())
best_model = sm.tsa.statespace.SARIMAX(data['curs'].dropna(axis=0), order=(3, d, 2), 

                                        seasonal_order=(3, D, 0, 12)).fit(disp=-1)

print(best_model.summary())

warnings.filterwarnings("ignore")
tsplot(best_model.resid[:], lags=30)
data['Arima'] = best_model.fittedvalues
data.head()
print('Количество недель в фактических данных:', len(data))
# Сделаем прогноз на основе предсказательной модели

forecast = best_model.predict(start = data.shape[0], end = data.shape[0]+52)

forecast = data['Arima'].append(forecast).values[-250:]

actual = data['curs'].values[-198:] #-146
# Визуализируем полученные данные

plt.figure(figsize=(15, 7))

plt.plot(forecast, color='r', label="model")

plt.title('Прогноз курса доллара на 52 недели 2020 г, SARIMA model')

plt.plot(actual, label="actual")

plt.legend()

plt.axvspan(len(actual), len(forecast), alpha=0.5, color='lightgrey')

plt.grid(True)