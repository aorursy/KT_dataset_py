import warnings                                  # `do not disturbe` mode

warnings.filterwarnings('ignore')



import numpy as np                               # vectors and matrices

import pandas as pd                              # tables and data manipulations

import matplotlib.pyplot as plt                  # plots

import seaborn as sns                            # more plots



from dateutil.relativedelta import relativedelta # working with dates with style

from scipy.optimize import minimize              # for function minimization



import statsmodels.formula.api as smf            # statistics and econometrics: 统计和计量经济学类库

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs



from itertools import product                    # some useful functions

from tqdm import tqdm_notebook



%matplotlib inline
# 加载时间序列数据：两个文件都只有一个时间列和一个数据列（分别Ads和GEMS_GEMS_SPEND），加载dataFrame时指定索引为Time并且按照日期类型来解析这一列

ads = pd.read_csv('../input/ads.csv', index_col=['Time'], parse_dates=['Time'])

currency = pd.read_csv('../input/currency.csv', index_col=['Time'], parse_dates=['Time'])

print(ads.columns)

print(currency.columns)
plt.figure(figsize=(15, 5))

plt.plot(ads.Ads)

plt.title('Ads watched (hourly data)')

plt.grid(True)

plt.show()
plt.figure(figsize=(15, 5))

plt.plot(currency.GEMS_GEMS_SPENT)

plt.title('In-game currency spent (daily data)')

plt.grid(True)

plt.show()
# 加载和定义上面提到的所有评估方法



# Importing everything from above

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error



def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# 计算移动均线上某一点取值的函数

def moving_average(series, n):

    """

        Calculate average of last n observations

    """

    return np.average(series[-n:])



moving_average(ads, 24) # prediction for the last observed day (past 24 hours)
# 绘制移动均线的函数，可选标注出异常值

def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):



    """

        series - dataframe with timeseries

        window - rolling window size 

        plot_intervals - show confidence intervals

        plot_anomalies - show anomalies 



    """

    # 移动均线数据

    rolling_mean = series.rolling(window=window).mean() 



    # 图片设置

    plt.figure(figsize=(15,5))

    plt.title("Moving average\n window size = {}".format(window))



    # 绘制移动均线

    plt.plot(rolling_mean, "g", label="Rolling mean trend")



    # 绘制置信区间

    # Plot confidence intervals for smoothed values

    if plot_intervals:

        # MAE

        mae = mean_absolute_error(series[window:], rolling_mean[window:]) 

        # 样本值到均线值的标准差

        deviation = np.std(series[window:] - rolling_mean[window:]) 

        # 下界：均线值 - 样本到均线的MAE(误差均值) - 样本到均线的标准差(误差波动)

        lower_bond = rolling_mean - (mae + scale * deviation) 

        # 上界：均线值 + 样本到均线的MAE(误差均值) + 样本到均线的标准差(误差波动)

        upper_bond = rolling_mean + (mae + scale * deviation) 

        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")

        plt.plot(lower_bond, "r--")

        

        # 标出异常值（在上界、下界以外的点）

        # Having the intervals, find abnormal values

        if plot_anomalies:

            anomalies = pd.DataFrame(index=series.index, columns=series.columns)

            anomalies[series<lower_bond] = series[series<lower_bond]

            anomalies[series>upper_bond] = series[series>upper_bond]

            plt.plot(anomalies, "ro", markersize=10)

    

    # 绘制移动均线

    plt.plot(series[window:], label="Actual values")

    plt.legend(loc="upper left")

    plt.grid(True)
# 4小时均线

plotMovingAverage(ads, 4) 
# 12小时均线，时间越长曲线越平滑

plotMovingAverage(ads, 12) 
# 24小时均线

plotMovingAverage(ads, 24)
# 绘制均线的置信区间

plotMovingAverage(ads, 4, plot_intervals=True)
# 构造一个异常值

ads_anomaly = ads.copy()

ads_anomaly.iloc[-20] = ads_anomaly.iloc[-20] * 0.2 # say we have 80% drop of ads 
# 绘制均线的置信区间，同时标出异常值

plotMovingAverage(ads_anomaly, 4, plot_intervals=True, plot_anomalies=True)
# 换一个数据集，绘制7天均线，可以发现有不少”异常值“，

# 这体现了移动均线的一个缺陷，以此图为例，它只能捕捉7天的趋势，不能捕捉月度/季度的数据Pattern

# 结果把每月例行的数据峰值当成了异常值

# 如果想要避免这类的错误，需要更复杂的模型

plotMovingAverage(currency, 7, plot_intervals=True, plot_anomalies=True) # weekly smoothing
# 计算加权移动均线上某一点取值的函数

def weighted_average(series, weights):

    """

        Calculate weighter average on series

    """

    result = 0.0

    weights.reverse()

    for n in range(len(weights)):

        result += series.iloc[-n-1] * weights[n]

    return float(result)
weighted_average(ads, [0.6, 0.3, 0.1])
def exponential_smoothing(series, alpha):

    """

        series - dataset with timestamps

        alpha - float [0.0, 1.0], smoothing parameter

    """

    result = [series[0]] # first value is same as series

    for n in range(1, len(series)):

        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    return result
def plotExponentialSmoothing(series, alphas):

    """

        Plots exponential smoothing with different alphas

        

        series - dataset with timestamps

        alphas - list of floats, smoothing parameters

        

    """

    with plt.style.context('seaborn-white'):    

        plt.figure(figsize=(15, 7)) 

        # 为每个alpha绘制一条平滑曲线

        for alpha in alphas:

            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))

        # 绘制原始时间序列

        plt.plot(series.values, "c", label = "Actual")

        # 图例、轴等

        plt.legend(loc="best")

        plt.axis('tight')

        plt.title("Exponential Smoothing")

        plt.grid(True);
# 可以看到：衰减系数越低，曲线越平滑；衰减系数越高，曲线越接近原始时间序列

plotExponentialSmoothing(ads.Ads, [0.3, 0.05])
plotExponentialSmoothing(currency.GEMS_GEMS_SPENT, [0.3, 0.05])
# Double Exponential Smoothing计算

def double_exponential_smoothing(series, alpha, beta):

    """

        series - dataset with timeseries

        alpha - float [0.0, 1.0], smoothing parameter for level

        beta - float [0.0, 1.0], smoothing parameter for trend

    """

    # first value is same as series

    result = [series[0]]

    for n in range(1, len(series)+1):

        if n == 1:

            level, trend = series[0], series[1] - series[0]

        if n >= len(series): # forecasting

            value = result[-1]

        else:

            value = series[n]

        last_level, level = level, alpha*value + (1-alpha)*(level+trend)

        trend = beta*(level-last_level) + (1-beta)*trend

        result.append(level+trend)

    return result



# 绘制Double Exponential Smoothing图

def plotDoubleExponentialSmoothing(series, alphas, betas):

    """

        Plots double exponential smoothing with different alphas and betas

        

        series - dataset with timestamps

        alphas - list of floats, smoothing parameters for level

        betas - list of floats, smoothing parameters for trend

    """

    

    with plt.style.context('seaborn-white'):    

        plt.figure(figsize=(20, 8))

        for alpha in alphas:

            for beta in betas:

                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))

        plt.plot(series.values, label = "Actual")

        plt.legend(loc="best")

        plt.axis('tight')

        plt.title("Double Exponential Smoothing")

        plt.grid(True)
plotDoubleExponentialSmoothing(ads.Ads, alphas=[0.9, 0.02], betas=[0.9, 0.02])
plotDoubleExponentialSmoothing(currency.GEMS_GEMS_SPENT, alphas=[0.9, 0.02], betas=[0.9, 0.02])
# HoltWinters代码，其中的初始化参数包括：

# * Series，slen：时间序列数据，时间序列的长度

# * alpha，beta，gamma：Holt-Winters模型参数，分别代表观测分量、趋势分量、周期分量对当前值（对应面是历史叠加值）的偏重程度，越高越偏重当前值

# * n_preds: prediction horizon

# * scaling_factor：Brutlag方法中置信区间宽度系数

class HoltWinters:

    

    """

    Holt-Winters model with the anomalies detection using Brutlag method

    

    # series - initial time series

    # slen - length of a season

    # alpha, beta, gamma - Holt-Winters model coefficients

    # n_preds - predictions horizon

    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)

    

    """

    

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):

        # 初始化HoltWinters及Brutlag的参数

        self.series = series

        self.slen = slen

        self.alpha = alpha

        self.beta = beta

        self.gamma = gamma

        self.n_preds = n_preds

        self.scaling_factor = scaling_factor

        

    def initial_trend(self):

        # 计算当前时刻的趋势分量

        sum = 0.0

        for i in range(self.slen):

            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen

        return sum / self.slen  

    

    def initial_seasonal_components(self):

        # 计算当前周期内每个时刻的周期分量

        seasonals = {}

        season_averages = []

        n_seasons = int(len(self.series)/self.slen)

        # let's calculate season averages

        for j in range(n_seasons):

            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))

        # let's calculate initial values

        for i in range(self.slen):

            sum_of_vals_over_avg = 0.0

            for j in range(n_seasons):

                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]

            seasonals[i] = sum_of_vals_over_avg/n_seasons

        return seasonals   



    def triple_exponential_smoothing(self):

        # 计算时间序列每个值（含历史时刻，以及n_preds个未来时刻）的平滑值，观测值分量，周期分量，趋势分量，预测偏差，置信区间上下届

        self.result = []

        self.Smooth = []

        self.Season = []

        self.Trend = []

        self.PredictedDeviation = []

        self.UpperBond = []

        self.LowerBond = []

        

        seasonals = self.initial_seasonal_components()

        

        for i in range(len(self.series)+self.n_preds):

            if i == 0: # components initialization

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

                

            if i >= len(self.series): # predicting

                m = i - len(self.series) + 1

                self.result.append((smooth + m*trend) + seasonals[i%self.slen])

                

                # when predicting we increase uncertainty on each step

                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 

                

            else:

                val = self.series[i]

                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)

                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend

                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]

                self.result.append(smooth+trend+seasonals[i%self.slen])

                

                # Deviation is calculated according to Brutlag algorithm.

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
# 时间序列数据的交叉验证类库

from sklearn.model_selection import TimeSeriesSplit # you have everything done for you



# MAE@“cross-validation on a rolling basis"

def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=24):

    """

        Returns error on CV  

        

        params - vector of parameters for optimization

        series - dataset with timeseries

        slen - season length for Holt-Winters model

    """

    # errors array

    errors = []

    

    values = series.values

    alpha, beta, gamma = params

    

    # set the number of folds for cross-validation

    tscv = TimeSeriesSplit(n_splits=3) 

    

    # iterating over folds, train model on each, forecast and calculate error

    for train, test in tscv.split(values):



        model = HoltWinters(series=values[train], slen=slen, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))

        model.triple_exponential_smoothing()

        

        predictions = model.result[-len(test):]

        actual = values[test]

        error = loss_function(predictions, actual)

        errors.append(error)

        

    return np.mean(np.array(errors))
%%time

from scipy.optimize import minimize



# 留一点数据用于预测

data = ads.Ads[:-20] # leave some data for testing

print(data.shape)



# 初始化模型的三个参数

# initializing model parameters alpha, beta and gamma

x = [0, 0, 0] 



# 最小化损失值：MAE@“cross-validation on a rolling basis"

# Minimizing the loss function

opt = minimize(timeseriesCVscore, x0=x, 

               args=(data, mean_squared_log_error), 

               method="TNC", 

               bounds = ((0, 1), (0, 1), (0, 1)) #值域约束条件

              )



# 得到最有的参数

# Take optimal values...

alpha_final, beta_final, gamma_final = opt.x

print(alpha_final, beta_final, gamma_final)



# 用最优参数初始化HoltWinters模型

# ...and train the model with them, forecasting for the next 50 hours

model = HoltWinters(data, slen = 24, 

                    alpha = alpha_final, 

                    beta = beta_final, 

                    gamma = gamma_final, 

                    n_preds = 50, scaling_factor = 3)



# 用模型计算平滑序列（存储在模型内部的成员变量中）

model.triple_exponential_smoothing()
# 绘制原始序列、平滑序列（训练时没有使用最后20个序列样本）

def plotHoltWinters(series, model=model, plot_intervals=False, plot_anomalies=False):

    """

        series - dataset with timeseries

        plot_intervals - show confidence intervals

        plot_anomalies - show anomalies 

    """

    # 绘制原始序列、平滑序列

    plt.figure(figsize=(20, 10))

    plt.plot(model.result,  label = "Model")

    plt.plot(series.values, label = "Actual")

    # 计算MAPE，打印在图片Title上

    plot_len = min(len(series.values), len(model.result))

    print(series.values.shape, len(model.result), model.result[:len(series)][0])

    error = mean_absolute_percentage_error(series.values[:plot_len], model.result[:plot_len])

    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

    # 标注异常值

    if plot_anomalies:

        anomalies = np.array([np.NaN]*plot_len)

        anomalies[series.values[:plot_len]<model.LowerBond[:plot_len]] = series.values[series.values[:plot_len]<model.LowerBond[:plot_len]]

        anomalies[series.values[:plot_len]>model.UpperBond[:plot_len]] = series.values[series.values[:plot_len]>model.UpperBond[:plot_len]]

        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

    # 标注平滑序列置信区间

    if plot_intervals:

        plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")

        plt.plot(model.LowerBond, "r--", alpha=0.5)

        plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.2, color = "grey")    

    # 用一根竖虚线标出原始序列样本的最大时刻

    plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')

    # 用灰色底色标出模型(平滑曲线)预测的部分（最后20个样本没有参与训练）

    plt.axvspan(len(series)-20, len(model.result), alpha=0.3, color='lightgrey')

    # 显示网格、设置轴样式、显示图例

    plt.grid(True)

    plt.axis('tight')

    plt.legend(loc="best", fontsize=13);
plotHoltWinters(ads.Ads, model)
plotHoltWinters(ads.Ads, model, plot_intervals=True, plot_anomalies=True)
plt.figure(figsize=(25, 5))

plt.plot(model.PredictedDeviation)

plt.grid(True)

plt.axis('tight')

plt.title("Brutlag's predicted deviation");
%%time

data = currency.GEMS_GEMS_SPENT[:-50] 

slen = 30 # 30-day seasonality



x = [0, 0, 0] 



opt = minimize(timeseriesCVscore, x0=x, 

               args=(data, mean_absolute_percentage_error, slen), 

               method="TNC", bounds = ((0, 1), (0, 1), (0, 1))

              )



alpha_final, beta_final, gamma_final = opt.x

print(alpha_final, beta_final, gamma_final)



model = HoltWinters(data, slen = slen, 

                    alpha = alpha_final, 

                    beta = beta_final, 

                    gamma = gamma_final, 

                    n_preds = 100, scaling_factor = 3)



model.triple_exponential_smoothing()



#import time

#time.sleep(120)
plotHoltWinters(currency.GEMS_GEMS_SPENT, model)
# 绘制HoltWinters平滑序列的置信区间及异常值

plotHoltWinters(currency.GEMS_GEMS_SPENT, model, plot_intervals=True, plot_anomalies=True)
# 绘制predicted deviation变化情况

plt.figure(figsize=(20, 5))

plt.plot(model.PredictedDeviation)

plt.grid(True)

plt.axis('tight')

plt.title("Brutlag's predicted deviation");
# 生成一份白噪声

white_noise = np.random.normal(size=1000)

with plt.style.context('bmh'):  

    plt.figure(figsize=(15, 5))

    plt.plot(white_noise)
# 将白噪声（是平稳时间序列）叠加在4个不同的时间序列上，查看叠加后的序列是否仍然是时间平稳的

# 标题顶部标注的是Dickey-Fuller p-value值，

#    该值代表着迪基-福勒检验作用在这个时间序列后，有多大可能存在一个单位根，即有多大可能是平稳时间序列

#    当p-value大于某个critical size，可以说这个时间序列被迪基-福勒检验拒绝，不是平稳时间序列

# 前3张图片的p-value是0，第4张为0.926

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm



def plotProcess(n_samples=1000, rho=0):

    x = w = np.random.normal(size=n_samples)

    for t in range(n_samples):

        x[t]  = rho * x[t-1] + w[t]



    with plt.style.context('bmh'):  

        plt.figure(figsize=(10, 3))

        plt.plot(x)

        adf_ret = sm.tsa.stattools.adfuller(x)

        print("---"*20)

        print("rho:\t", rho)

        print("adf:\t",adf_ret[0])

        print("pvalue:\t",adf_ret[1])

        print("usedlag:\t",adf_ret[2])

        print("critical values:\t",adf_ret[4])

        plt.title("Rho {}\n Dickey-Fuller p-value: {}".format(rho, round(adf_ret[1], 3)))

        

for rho in [0, 0.6, 0.9, 1]:

    plotProcess(rho=rho)
import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm





# 下面处理步骤用到的一个公用函数 tsplot(Time Series Plot)

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):

    """

        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        

        y - timeseries

        lags - how many lags to include in ACF, PACF calculation

    """

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

        

    with plt.style.context(style):    

        fig      = plt.figure(figsize=figsize)

        layout   = (2, 2)

        ts_ax    = plt.subplot2grid(layout, (0, 0), colspan=2)  #图1（顶部）：时间序列(y)，以及标题中标注的Dickey-Fuller:p=xxx

        acf_ax   = plt.subplot2grid(layout, (1, 0))             #图2（左下）：ACF图

        pacf_ax  = plt.subplot2grid(layout, (1, 1))             #图3（右下）：PACF图

        

        # 图1：计算迪基-福勒检验的p-value，它的值大于某个critical size时代表着被迪基-福勒检验拒绝，不是平稳时间序列

        y.plot(ax=ts_ax)

        p_value = sm.tsa.stattools.adfuller(y)[1]

        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))

        # 图2：ACF， 横轴lags从0到60，纵轴correlations(相关系数：协方差/各自标准差乘积)

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)

        # 图3：PACF，横轴lags从0到60，纵轴partial autocorrelation(偏相关系数)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)

        plt.tight_layout()
tsplot(ads.Ads, lags=60)
# 周期长度24，减去一个周期前的值

ads_diff = ads.Ads - ads.Ads.shift(24)

tsplot(ads_diff[24:], lags=60)
# 使用First Difference再得到一个序列

ads_diff = ads_diff - ads_diff.shift(1)

tsplot(ads_diff[24+1:], lags=60)
tsplot(ads_diff[24+1:], lags=60)
# 一组参数，可以设定具体值（如是否开启First Difference的两个参数），也可以设定参数搜索范围(根据前面的方法，观测ACF、PACF图)

# setting initial values and some bounds for them

ps = range(2, 5)  #p: 当前序列值取决于之前几个lag

d=1               #d: 当前序列值计算是否使用First Difference

qs = range(2, 5)  #q: 当前误差值取决于之前几个lag

Ps = range(0, 2)  #P: 周期因子取决于之前几个周期

D=1               #D: 周期因子计算是否使用First Difference

Qs = range(0, 2)  #Q: 周期因子误差取决于之前几个周期

s  = 24 # season length is still 24



# 计算所有的参数组合，共36种

# creating list with all the possible combinations of parameters

parameters = product(ps, qs, Ps, Qs)

parameters_list = list(parameters)

print(parameters_list)

len(parameters_list)
from itertools import product

from tqdm import tqdm_notebook



# 前面对数据做各种预处理、得到比较合适的ACF，PACF图，只是为了找到模型出超参数的范围

# 现在将这些模型超参数（paramters_list)列表喂给SARIMA模型，由模型来替代之前手动做的那些操作，拟合时间序列

def optimizeSARIMA(parameters_list, d, D, s):

    """

        Return dataframe with parameters and corresponding AIC

        

        parameters_list - list with (p, q, P, Q) tuples

        d - integration order in ARIMA model

        D - seasonal integration order 

        s - length of season

    """    

    results  = []

    best_aic = float("inf")



    for param in tqdm_notebook(parameters_list):

        # 有些参数下，模型不能收敛，会超时抛出异常

        # we need try-except because on some combinations model fails to converge

        try:

            # 训练SARIMAX模型，返回结果类型为MLEResults

            model=sm.tsa.statespace.SARIMAX(ads.Ads,                                   # 样本

                                            order=(param[0], d, param[1]),             # 序列值相关的3个超参数：p，d，q

                                            seasonal_order=(param[2], D, param[3], s)  # 周期相关的超参数：P，D，Q，s

                                           ).fit(disp=-1)                              # disp=-1不打印收敛信息

        except:

            continue

        # 赤池信息量准则(Akaike information criterion )

        aic = model.aic

        # 记录AIX分数最高的模型参数(saving best model, AIC and parameters)

        if aic < best_aic:

            best_model = model

            best_aic = aic

            best_param = param

        # [(param, aic), (param, aic), ...]

        results.append([param, model.aic])

    

    # 统计各版本模型的AIC指标，AIC越低代表模型效果越好

    # sorting in ascending order, the lower AIC is - the better    

    result_table = pd.DataFrame(results)

    result_table.columns = ['parameters', 'aic']

    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table
%%time

result_table = optimizeSARIMA(parameters_list, d, D, s)
# 模型分数

result_table.head()
# 用最优参数训练SARIMAX模型，打印模型summary

# set the parameters that give the lowest AIC

p, q, P, Q = result_table.parameters[0]



# 用上一步search到的最优参数训练SARIMAX模型，返回结果类型为MLEResults

best_model=sm.tsa.statespace.SARIMAX(ads.Ads, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)

print(best_model.summary())
# 查看模型的Dickey-Fuller pvalue，ACF图，PACF图

tsplot(best_model.resid[24+1:], lags=60)
# 用模型对时间序列做预测

def plotSARIMA(series, model, n_steps):

    """

        Plots model vs predicted values

        

        series - dataset with timeseries

        model - fitted SARIMA model

        n_steps - number of steps to predict in the future

        

    """

    # adding model values

    data = series.copy()

    data.columns = ['actual']

    data['arima_model'] = model.fittedvalues

    # making a shift on s+d steps, because these values were unobserved by the model

    # due to the differentiating

    data['arima_model'][:s+d] = np.NaN

    

    # forecasting on n_steps forward 

    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)

    forecast = data.arima_model.append(forecast)

    # calculate error, again having shifted on s+d steps from the beginning

    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])



    plt.figure(figsize=(15, 7))

    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

    plt.plot(forecast, color='r', label="model")

    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')

    plt.plot(data.actual, label="actual")

    plt.legend()

    plt.grid(True);
plotSARIMA(ads, best_model, 50)
# 样本Label

# Creating a copy of the initial datagrame to make various transformations 

data = pd.DataFrame(ads.Ads.copy())

data.columns = ["y"]
# 用Lags of Time Sereis来构造的特征

# Adding the lag of the target variable from 6 steps back up to 24

for i in range(6, 25):

    data["lag_{}".format(i)] = data.y.shift(i)
# 查看特征

# take a look at the new dataframe 

data.tail(n=5)
# 用前面的编写的函数，来对时间序列做交叉验证

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score



# for time-series cross-validation set 5 folds 

tscv = TimeSeriesSplit(n_splits=5)
# 拆分训练集和测试集，前一段为训练集，后一段为测试集，test_size表示测试集样本的占比

def timeseries_train_test_split(X, y, test_size):

    """

        Perform train-test split with respect to time series structure

    """

    

    # get the index after which test set starts

    test_index = int(len(X)*(1-test_size))

    

    X_train = X.iloc[:test_index]

    y_train = y.iloc[:test_index]

    X_test = X.iloc[test_index:]

    y_test = y.iloc[test_index:]

    

    return X_train, X_test, y_train, y_test



y = data.dropna().y

X = data.dropna().drop(['y'], axis=1)



# reserve 30% of data for testing

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
# 训练模型

# machine learning in two lines

lr = LinearRegression()

lr.fit(X_train, y_train)
# 可视化预测结果

def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):

    """

        Plots modelled vs fact values, prediction intervals and anomalies

    

    """

    # 预测

    prediction = model.predict(X_test)

    

    # 绘制两条曲线：样本标签，样本预测值

    plt.figure(figsize=(15, 7))

    plt.plot(prediction, "g", label="prediction", linewidth=2.0)

    plt.plot(y_test.values, label="actual", linewidth=2.0)

    

    if plot_intervals:

        # 交叉验证MAE的均值和标准差

        cv = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")

        mae = cv.mean() * (-1)

        deviation = cv.std()

        # 计算lower/upper bound：预测值上下(MAE + 1.96*标准差)

        scale = 1.96

        lower = prediction - (mae + scale * deviation)

        upper = prediction + (mae + scale * deviation)

        # 绘制lower/upper bound

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)

        plt.plot(upper, "r--", alpha=0.5)

        # 标注异常值

        if plot_anomalies:

            anomalies = np.array([np.NaN]*len(y_test))

            anomalies[y_test<lower] = y_test[y_test<lower]

            anomalies[y_test>upper] = y_test[y_test>upper]

            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

    

    # 计算MAPE，注释在title中

    error = mean_absolute_percentage_error(prediction, y_test)

    plt.title("Mean absolute percentage error {0:.2f}%".format(error))

    # 绘图

    plt.legend(loc="best")

    plt.tight_layout()

    plt.grid(True);



plotModelResults(lr, plot_intervals=True)
# 用柱状图绘制线性模型的参数：因为每个特征对应一个lag，因此每个参数也对应某个lag在线性回归中的权重

def plotCoefficients(model):

    """

        Plots sorted coefficient values of the model

    """

    

    coefs = pd.DataFrame(model.coef_, X_train.columns)

    coefs.columns = ["coef"]

    coefs["abs"] = coefs.coef.apply(np.abs)

    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    

    plt.figure(figsize=(15, 7))

    coefs.coef.plot(kind='bar')

    plt.grid(True, axis='y')

    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');



plotCoefficients(lr)
# data.index是时间，可以从中提取出hour和weekday

print(data.index[:3])



# 转成时间格式

data.index = pd.to_datetime(data.index)



# 提取hour, weekday, is_weekend

data["hour"] = data.index.hour

data["weekday"] = data.index.weekday

data['is_weekend'] = data.weekday.isin([5,6])*1

data.tail()
# 查看这些特征的取值

plt.figure(figsize=(16, 5))

plt.title("Encoded features")

data.hour.plot()

data.weekday.plot()

data.is_weekend.plot()

plt.grid(True);
# 用StandardScaler来scale这些特征的值

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# 加入新特征后重新训练模型，可视化预测结果

y = data.dropna().y

X = data.dropna().drop(['y'], axis=1)



X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled  = scaler.transform(X_test)



lr = LinearRegression()

lr.fit(X_train_scaled, y_train)



plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)

plotCoefficients(lr)
# 均值编码用的函数：对data中的样本，根据cat_feature对real_feature分组求均值，生成新的特征

def code_mean(data, cat_feature, real_feature):

    """

    Returns a dictionary where keys are unique categories of the cat_feature,

    and values are means over real_feature

    """

    return dict(data.groupby(cat_feature)[real_feature].mean())
# 根据训练集（不能包含验证集和测试集）的hour对y求均值生成新的特征，查看这些特征的效果

average_hour = code_mean(data, 'hour', "y")

plt.figure(figsize=(7, 5))

plt.title("Hour averages")

pd.DataFrame.from_dict(average_hour, orient='index')[0].plot()

plt.grid(True);
# 写一个函数，将前面的Lag of Time Series、Date and time features、Taget Encoding结合在一起，生成多种特征

def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):

    """

        series: pd.DataFrame

            dataframe with timeseries



        lag_start: int

            initial step back in time to slice target variable 

            example - lag_start = 1 means that the model 

                      will see yesterday's values to predict today



        lag_end: int

            final step back in time to slice target variable

            example - lag_end = 4 means that the model 

                      will see up to 4 days back in time to predict today



        test_size: float

            size of the test dataset after train/test split as percentage of dataset



        target_encoding: boolean

            if True - add target averages to the dataset

        

    """

    

    # copy of the initial dataset

    data = pd.DataFrame(series.copy())

    data.columns = ["y"]

    

    # lags of series

    for i in range(lag_start, lag_end):

        data["lag_{}".format(i)] = data.y.shift(i)

    

    # datetime features

    data.index = pd.to_datetime(data.index)

    data["hour"] = data.index.hour

    data["weekday"] = data.index.weekday

    data['is_weekend'] = data.weekday.isin([5,6])*1

    

    # 用target_encoding替代weekday_average, hour_average

    if target_encoding:

        # calculate averages on train set only

        test_index = int(len(data.dropna())*(1-test_size))

        data['weekday_average'] = list(map(code_mean(data[:test_index], 'weekday', "y").get, data.weekday))

        data["hour_average"] = list(map(code_mean(data[:test_index], 'hour', "y").get, data.hour))



        # frop encoded variables 

        data.drop(["hour", "weekday"], axis=1, inplace=True)

    

    # train-test split

    y = data.dropna().y

    X = data.dropna().drop(['y'], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)



    return X_train, X_test, y_train, y_test
# 重新训练模型，可视化模型预测效果，特征重要程度

X_train, X_test, y_train, y_test = prepareData(ads.Ads, lag_start=6, lag_end=25, test_size=0.3, target_encoding=True)



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



lr = LinearRegression()

lr.fit(X_train_scaled, y_train)



plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)

plotCoefficients(lr)
# 关闭Target Encoding再训练一次

X_train, X_test, y_train, y_test = prepareData(ads.Ads, lag_start=6, lag_end=25, test_size=0.3, target_encoding=False)



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
plt.figure(figsize=(10, 8))

sns.heatmap(X_train.corr());
from sklearn.linear_model import LassoCV, RidgeCV



ridge = RidgeCV(cv=tscv)

ridge.fit(X_train_scaled, y_train)



plotModelResults(ridge, 

                 X_train=X_train_scaled, 

                 X_test=X_test_scaled, 

                 plot_intervals=True, plot_anomalies=True)

plotCoefficients(ridge)
lasso = LassoCV(cv=tscv)

lasso.fit(X_train_scaled, y_train)



plotModelResults(lasso, 

                 X_train=X_train_scaled, 

                 X_test=X_test_scaled, 

                 plot_intervals=True, plot_anomalies=True)

plotCoefficients(lasso)
from xgboost import XGBRegressor 



xgb = XGBRegressor()

xgb.fit(X_train_scaled, y_train)
plotModelResults(xgb, 

                 X_train=X_train_scaled, 

                 X_test=X_test_scaled, 

                 plot_intervals=True, plot_anomalies=True)