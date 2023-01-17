import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install arch
import sys

import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm



import pandas as pd

import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error



import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs

from scipy.optimize import minimize

from arch import arch_model



import matplotlib.pyplot as plt

import matplotlib as mpl

import matplotlib.dates as mdates

%matplotlib inline





from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import graph_objs as go

init_notebook_mode(connected = True)



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
dataset = pd.read_csv('/kaggle/input/hour-online/hour_online.csv', index_col=['Time'], parse_dates=['Time'])

plotly_df(dataset, title = "Online users")
def moving_average(series, n):

    return np.average(series[-n:])



moving_average(dataset.Users, 24) # посмотрим на прогноз, построенный по последнему наблюдаемому дню (24 часа)
def plotMovingAverage(series, n, plot_bounds=False):

    

    """

    series - dataframe with timeseries

    n - rolling window size 

    plot_bounds: bool - whether to draw confidence interval

    """

    

    rolling_mean = series.rolling(window=n).mean()

        

    fig, ax = plt.subplots(figsize=(15,5))

    plt.title("Moving average\n window size = {}".format(n))

    plt.plot(rolling_mean, "g", label="Rolling mean trend")



    # При желании, можно строить и доверительные интервалы для сглаженных значений

    if plot_bounds:

        rolling_std =  series.rolling(window=n).std()

        upper_bound = rolling_mean+1.96*rolling_std

        lower_bound = rolling_mean-1.96*rolling_std

        plt.plot(upper_bound, "r--", label="Upper Bound / Lower Bound")

        plt.plot(lower_bound, "r--")

    plt.plot(series[n:], label="Actual values")

    plt.legend(loc="upper left")

    plt.grid(True)

    # Деления соответствуют понедельникам что дает представление о недельной цикличности графика

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))

    # Отображать значение дат в формате yy-mm-dd

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'));

    plt.xticks(rotation=45)
plotMovingAverage(dataset, 24) # сглаживаем по дням
plotMovingAverage(dataset, 24*7, plot_bounds=True) # сглаживаем по неделям
plotMovingAverage(dataset, 24*7*4, plot_bounds=True) # сглаживаем по месяцам
def weighted_average(series, weights):

    result = 0.0

    weights.reverse()

    for n in range(len(weights)):

        result += series[-n-1] * weights[n]

    return result
weighted_average(dataset.Users, [0.6, 0.2, 0.1, 0.07, 0.03])
def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series

    for n in range(1, len(series)):

        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    return result
with plt.style.context('seaborn-white'):    

    plt.figure(figsize=(20, 8))

    for alpha in [0.3, 0.05]:

        plt.plot(exponential_smoothing(dataset.Users, alpha), label="Alpha {}".format(alpha))

    plt.plot(dataset.Users.values, "c", label = "Actual")

    plt.legend(loc="best")

    plt.axis('tight')

    plt.title("Exponential Smoothing")

    plt.grid(True)
def double_exponential_smoothing(series, alpha, beta):

    result = [series[0]]

    for n in range(1, len(series)+1):

        if n == 1:

            level, trend = series[0], series[1] - series[0]

        if n >= len(series): # прогнозируем

            value = result[-1]

        else:

            value = series[n]

        last_level, level = level, alpha*value + (1-alpha)*(level+trend)

        trend = beta*(level-last_level) + (1-beta)*trend

        result.append(level+trend)

    return result





with plt.style.context('seaborn-white'):    

    plt.figure(figsize=(20, 8))

    for alpha in [0.9, 0.02]:

        for beta in [0.9, 0.02]:

            plt.plot(double_exponential_smoothing(dataset.Users, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))

    plt.plot(dataset.Users.values, label = "Actual")

    plt.legend(loc="best")

    plt.axis('tight')

    plt.title("Double Exponential Smoothing")

    plt.grid(True)
class HoltWinters:

    

    """

    Модель Хольта-Винтерса с методом Брутлага для детектирования аномалий

    https://fedcsis.org/proceedings/2012/pliks/118.pdf

    

    

    # series - исходный временной ряд

    # slen - длина сезона

    # alpha, beta, gamma - коэффициенты модели Хольта-Винтерса

    # n_preds - горизонт предсказаний

    # scaling_factor - задаёт ширину доверительного интервала по Брутлагу (обычно принимает значения от 2 до 3)

    

    """

    

    

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):

        self.series = series

        self.slen = slen

        self.alpha = alpha

        self.beta = beta

        self.gamma = gamma

        self.n_preds = n_preds

        self.scaling_factor = scaling_factor

        

        

    def initial_trend(self):

        sum = 0.0

        for i in range(self.slen):

            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen

        return sum / self.slen  

    

    def initial_seasonal_components(self):

        seasonals = {}

        season_averages = []

        n_seasons = int(len(self.series)/self.slen)

        # вычисляем сезонные средние

        for j in range(n_seasons):

            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))

        # вычисляем начальные значения

        for i in range(self.slen):

            sum_of_vals_over_avg = 0.0

            for j in range(n_seasons):

                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]

            seasonals[i] = sum_of_vals_over_avg/n_seasons

        return seasonals   



          

    def triple_exponential_smoothing(self):

        self.result = []

        self.Smooth = []

        self.Season = []

        self.Trend = []

        self.PredictedDeviation = []

        self.UpperBond = []

        self.LowerBond = []

        

        seasonals = self.initial_seasonal_components()

        

        for i in range(len(self.series)+self.n_preds):

            if i == 0: # инициализируем значения компонент

                smooth = self.series[0]

                trend = self.initial_trend()

                self.result.append(self.series[0])

                self.Smooth.append(smooth)

                self.Trend.append(trend)

                self.Season.append(seasonals[i%self.slen])

                

                self.PredictedDeviation.append(0)

                

                self.UpperBond.append(self.result[0] + 

                                      self.scaling_factor * 

                                      self.PredictedDeviation[0])

                

                

                self.LowerBond.append(self.result[0] - 

                                      self.scaling_factor * 

                                      self.PredictedDeviation[0])

                

                

                

                continue

            if i >= len(self.series): # прогнозируем

                m = i - len(self.series) + 1

                self.result.append((smooth + m*trend) + seasonals[i%self.slen])

                

                # во время прогноза с каждым шагом увеличиваем неопределенность

                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 

                

            else:

                val = self.series[i]

                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)

                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend

                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]

                self.result.append(smooth+trend+seasonals[i%self.slen])

                

                # Отклонение рассчитывается в соответствии с алгоритмом Брутлага

                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 

                                               + (1-self.gamma)*self.PredictedDeviation[-1])

                

            

            self.UpperBond.append(self.result[-1] + 

                                  self.scaling_factor * 

                                  self.PredictedDeviation[-1])





            self.LowerBond.append(self.result[-1] - 

                                  self.scaling_factor * 

                                  self.PredictedDeviation[-1])

                

                

                

            

            self.Smooth.append(smooth)

            self.Trend.append(trend)

            self.Season.append(seasonals[i%self.slen])

            

            

        
from sklearn.model_selection import TimeSeriesSplit



def timeseriesCVscore(x):

    # вектор ошибок

    errors = []

    

    values = data.values

    alpha, beta, gamma = x

    

    # задаём число фолдов для кросс-валидации

    tscv = TimeSeriesSplit(n_splits=3) 

    

    # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку

    for train, test in tscv.split(values):



        model = HoltWinters(series=values[train], slen = 24*7, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))

        model.triple_exponential_smoothing()

        

        predictions = model.result[-len(test):]

        actual = values[test]

        error = mean_squared_error(predictions, actual)

        errors.append(error)

    return np.mean(np.array(errors))
%%time

data = dataset.Users[:-500] # отложим часть данных для тестирования



# инициализируем значения параметров

x = [0, 0, 0] 



# Минимизируем функцию потерь с ограничениями на параметры

opt = minimize(timeseriesCVscore, x0=x, method="TNC", bounds = ((0, 1), (0, 1), (0, 1)))



# Из оптимизатора берем оптимальное значение параметров

alpha_final, beta_final, gamma_final = opt.x

print(alpha_final, beta_final, gamma_final)
# Передаем оптимальные значения модели, 

data = dataset.Users

model = HoltWinters(data[:-128], slen = 24*7, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 128, scaling_factor = 2.56)

model.triple_exponential_smoothing()
def plotHoltWinters():

    Anomalies = np.array([np.NaN]*len(data))

    Anomalies[data.values<model.LowerBond] = data.values[data.values<model.LowerBond]

    plt.figure(figsize=(25, 10))

    plt.plot(model.result, label = "Model")

    plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")

    plt.plot(model.LowerBond, "r--", alpha=0.5)

    plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.5, color = "grey")

    plt.plot(data.values, label = "Actual")

    plt.plot(Anomalies, "o", markersize=10, label = "Anomalies")

    plt.axvspan(len(data)-128, len(data), alpha=0.5, color='lightgrey')

    plt.grid(True)

    plt.axis('tight')

    plt.legend(loc="best", fontsize=13);
plotHoltWinters()
plt.figure(figsize=(25, 5))

plt.plot(model.PredictedDeviation)

plt.grid(True)

plt.axis('tight')

plt.title("Brutlag's predicted deviation");
white_noise = np.random.normal(size=1000)

with plt.style.context('bmh'):  

    plt.figure(figsize=(15, 5))

    plt.plot(white_noise)
def plotProcess(n_samples=1000, rho=0):

    x = w = np.random.normal(size=n_samples)

    for t in range(n_samples):

        x[t] = rho * x[t-1] + w[t]



    with plt.style.context('bmh'):  

        plt.figure(figsize=(10, 3))

        plt.plot(x)

        plt.title("Rho {}\n Dickey-Fuller p-value: {}".format(rho, round(sm.tsa.stattools.adfuller(x)[1], 3)))

        

for rho in [0, 0.6, 0.9, 1]:

    plotProcess(rho=rho)
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

        ts_ax.set_title('Time Series Analysis Plots')

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)



        print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(y)[1])





        plt.tight_layout()

    return 
tsplot(dataset.Users, lags=30)
def invboxcox(y,lmbda):

    # обратное преобразование Бокса-Кокса

    if lmbda == 0:

        return(np.exp(y))

    else:

        return(np.exp(np.log(lmbda*y+1)/lmbda))





data = dataset.copy()

data['Users_box'], lmbda = scs.boxcox(data.Users+1) # прибавляем единицу, так как в исходном ряде есть нули

tsplot(data.Users_box, lags=30)

print("Оптимальный параметр преобразования Бокса-Кокса: %f" % lmbda)
data['Users_box_season'] = data.Users_box - data.Users_box.shift(24*7)

tsplot(data.Users_box_season[24*7:], lags=30)
data['Users_box_season_diff'] = data.Users_box_season - data.Users_box_season.shift(1)

tsplot(data.Users_box_season_diff[24*7+1:], lags=30)
ps = range(0, 5)

d=1

qs = range(0, 4)

Ps = range(0, 5)

D=1

Qs = range(0, 1)



from itertools import product



parameters = product(ps, qs, Ps, Qs)

parameters_list = list(parameters)

len(parameters_list)
%%time

results = []

best_aic = float("inf")

warnings.filterwarnings('ignore')



for param in parameters_list:

    #try except нужен, потому что на некоторых наборах параметров модель не обучается

    try:

        model=sm.tsa.statespace.SARIMAX(data.Users_box, order=(param[0], d, param[1]), 

                                        seasonal_order=(param[3], D, param[3], 24)).fit(disp=-1)

    #выводим параметры, на которых модель не обучается и переходим к следующему набору

    except ValueError:

        #print('wrong parameters:', param)

        continue

    aic = model.aic

    #сохраняем лучшую модель, aic, параметры

    if aic < best_aic:

        best_model = model

        best_aic = aic

        best_param = param

    results.append([param, model.aic])

    

warnings.filterwarnings('default')



result_table = pd.DataFrame(results)

result_table.columns = ['parameters', 'aic']

# print(result_table.sort_values(by = 'aic', ascending=True).head())
%%time

best_model=sm.tsa.statespace.SARIMAX(data.Users_box, order=(4, d, 3), 

                                        seasonal_order=(4, D, 1, 24)).fit(disp=-1)
print(best_model.summary())
tsplot(best_model.resid[24:], lags=30)
data["arima_model"] = invboxcox(best_model.fittedvalues, lmbda)

forecast = invboxcox(best_model.predict(start = data.shape[0], end = data.shape[0]+100), lmbda)

forecast = data.arima_model.append(forecast).values[-500:]

actual = data.Users.values[-400:]

plt.figure(figsize=(15, 7))

plt.plot(forecast, color='r', label="model")

plt.title("SARIMA model\n Mean absolute error {} users".format(round(mean_absolute_error(data.dropna().Users, data.dropna().arima_model))))

plt.plot(actual, label="actual")

plt.legend()

plt.axvspan(len(actual), len(forecast), alpha=0.5, color='lightgrey')

plt.grid(True)
def code_mean(data, cat_feature, real_feature):

    """

    Возвращает словарь, где ключами являются уникальные категории признака cat_feature, 

    а значениями - средние по real_feature

    """

    return dict(data.groupby(cat_feature)[real_feature].mean())
data = pd.DataFrame(dataset.copy())

data.columns = ["y"]



data.index = pd.to_datetime(data.index)

data["hour"] = data.index.hour

data["weekday"] = data.index.weekday

data['is_weekend'] = data.weekday.isin([5,6])*1

data.head()
code_mean(data, 'weekday', "y")
def prepareData(data, lag_start=5, lag_end=20, test_size=0.15):

    

    data = pd.DataFrame(data.copy())

    data.columns = ["y"]

    

    # считаем индекс в датафрейме, после которого начинается тестовый отрезок

    test_index = int(len(data)*(1-test_size))

    

    # добавляем лаги исходного ряда в качестве признаков

    for i in range(lag_start, lag_end):

        data["lag_{}".format(i)] = data.y.shift(i)

        

    

    data.index = pd.to_datetime(data.index)

    data["hour"] = data.index.hour

    data["weekday"] = data.index.weekday

    data['is_weekend'] = data.weekday.isin([5,6])*1

    

    # считаем средние только по тренировочной части, чтобы избежать лика

    data['weekday_average'] = list(map(code_mean(data[:test_index], 'weekday', "y").get, data.weekday))

    data["hour_average"] = list(map(code_mean(data[:test_index], 'hour', "y").get, data.hour))



    # выкидываем закодированные средними признаки 

    data.drop(["hour", "weekday"], axis=1, inplace=True)



    data = data.dropna()

    data = data.reset_index(drop=True)

    

    

    # разбиваем весь датасет на тренировочную и тестовую выборку

    X_train = data.loc[:test_index].drop(["y"], axis=1)

    y_train = data.loc[:test_index]["y"]

    X_test = data.loc[test_index:].drop(["y"], axis=1)

    y_test = data.loc[test_index:]["y"]

    

    return X_train, X_test, y_train, y_test
from sklearn.linear_model import LinearRegression



X_train, X_test, y_train, y_test = prepareData(dataset.Users, test_size=0.3, lag_start=12, lag_end=48)

lr = LinearRegression()

lr.fit(X_train, y_train)

prediction = lr.predict(X_test)

plt.figure(figsize=(15, 7))

plt.plot(prediction, "r", label="prediction")

plt.plot(y_test.values, label="actual")

plt.legend(loc="best")

plt.title("Linear regression\n Mean absolute error {} users".format(round(mean_absolute_error(prediction, y_test))))

plt.grid(True);
def performTimeSeriesCV(X_train, y_train, number_folds, model, metrics):

    print('Size train set: {}'.format(X_train.shape))

    

    k = int(np.floor(float(X_train.shape[0]) / number_folds))

    print('Size of each fold: {}'.format(k))



    errors = np.zeros(number_folds-1)



    # loop from the first 2 folds to the total number of folds    

    for i in range(2, number_folds + 1):

        print()

        split = float(i-1)/i

        print('Splitting the first ' + str(i) + ' chunks at ' + str(i-1) + '/' + str(i) )



        X = X_train[:(k*i)]

        y = y_train[:(k*i)]

        print('Size of train + test: {}'.format(X.shape)) # the size of the dataframe is going to be k*i



        index = int(np.floor(X.shape[0] * split))

        

        # folds used to train the model        

        X_trainFolds = X[:index]        

        y_trainFolds = y[:index]

        

        # fold used to test the model

        X_testFold = X[(index + 1):]

        y_testFold = y[(index + 1):]

        

        model.fit(X_trainFolds, y_trainFolds)

        errors[i-2] = metrics(model.predict(X_testFold), y_testFold)



    

    # the function returns the mean of the errors on the n-1 folds    

    return errors.mean()
performTimeSeriesCV(X_train, y_train, 5, lr, mean_absolute_error)
sys.path.append('/Users/dmitrys/xgboost/python-package/')

import xgboost as xgb



def XGB_forecast(data, lag_start=5, lag_end=20, test_size=0.15, scale=1.96):

    

    # исходные данные

    X_train, X_test, y_train, y_test = prepareData(dataset.Users, lag_start, lag_end, test_size)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    dtest = xgb.DMatrix(X_test)

    

    # задаём параметры

    params = {

        'objective': 'reg:linear',

        'booster':'gblinear'

    }

    trees = 1000

    

    # прогоняем на кросс-валидации с метрикой rmse

    cv = xgb.cv(params, dtrain, metrics = ('rmse'), verbose_eval=False, nfold=10, show_stdv=False, num_boost_round=trees)

    

    # обучаем xgboost с оптимальным числом деревьев, подобранным на кросс-валидации

    bst = xgb.train(params, dtrain, num_boost_round=cv['test-rmse-mean'].argmin())

    

    # можно построить кривые валидации

    #cv.plot(y=['test-mae-mean', 'train-mae-mean'])

    

    # запоминаем ошибку на кросс-валидации

    deviation = cv.loc[cv['test-rmse-mean'].argmin()]["test-rmse-mean"]

    

    # посмотрим, как модель вела себя на тренировочном отрезке ряда

    prediction_train = bst.predict(dtrain)

    plt.figure(figsize=(15, 5))

    plt.plot(prediction_train)

    plt.plot(y_train)

    plt.axis('tight')

    plt.grid(True)

    

    # и на тестовом

    prediction_test = bst.predict(dtest)

    lower = prediction_test-scale*deviation

    upper = prediction_test+scale*deviation



    Anomalies = np.array([np.NaN]*len(y_test))

    Anomalies[y_test<lower] = y_test[y_test<lower]



    plt.figure(figsize=(15, 5))

    plt.plot(prediction_test, label="prediction")

    plt.plot(lower, "r--", label="upper bond / lower bond")

    plt.plot(upper, "r--")

    plt.plot(list(y_test), label="y_test")

    plt.plot(Anomalies, "ro", markersize=10)

    plt.legend(loc="best")

    plt.axis('tight')

    plt.title("XGBoost Mean absolute error {} users".format(round(mean_absolute_error(prediction_test, y_test))))

    plt.grid(True)

    plt.legend()
XGB_forecast(dataset, test_size=0.2, lag_start=12, lag_end=48)