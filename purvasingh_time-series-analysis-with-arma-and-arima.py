from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.stattools import acf



import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np



sns.set()

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (14, 8)
# Draw samples from a standard Normal distribution (mean=0, stdev=1).

points = np.random.standard_normal(1000)



# making starting point as 0

points[0]=0



# Return the cumulative sum of the elements along a given axis.

random_walk = np.cumsum(points)

random_walk_series = pd.Series(random_walk)
plt.figure(figsize=[10, 7.5]); # Set dimensions for figure

plt.plot(random_walk)

plt.title("Simulated Random Walk")

plt.show()
random_walk_acf = acf(random_walk)

acf_plot = plot_acf(random_walk_acf, lags=20)
random_walk_difference = np.diff(random_walk, n=1)



plt.figure(figsize=[10, 7.5]); # Set dimensions for figure

plt.plot(random_walk_difference)

plt.title('Noise')

plt.show()
cof_plot_difference = plot_acf(random_walk_difference, lags=20);
from statsmodels.tsa.arima_process import ArmaProcess



# start by specifying the lag

ar3 = np.array([3])



# specify the weights : [1, 0.9, 0.3, -0.2]

ma3 = np.array([1, 0.9, 0.3, -0.2])



# simulate the process and generate 1000 data points

MA_3_process = ArmaProcess(ar3, ma3).generate_sample(nsample=1000)
plt.figure(figsize=[10, 7.5]); # Set dimensions for figure

plt.plot(MA_3_process)

plt.title('Simulation of MA(3) Model')

plt.show()

plot_acf(MA_3_process, lags=20);
ar3 = np.array([1, 0.9, 0.3, -0.2])

ma = np.array([3])

simulated_ar3_points = ArmaProcess(ar3, ma).generate_sample(nsample=10000)
plt.figure(figsize=[10, 7.5]); # Set dimensions for figure

plt.plot(simulated_ar3_points)

plt.title("Simulation of AR(3) Process")

plt.show()
plot_acf(simulated_ar3_points);
from statsmodels.graphics.tsaplots import plot_pacf



plot_pacf(simulated_ar3_points);
from statsmodels.tsa.stattools import pacf



pacf_coef_AR3 = pacf(simulated_ar3_points)

print(pacf_coef_AR3)
ar1 = np.array([1, 0.6])

ma1 = np.array([1, -0.2])

simulated_ARMA_1_1_points = ArmaProcess(ar1, ma1).generate_sample(nsample=10000)
plt.figure(figsize=[15, 7.5]); # Set dimensions for figure

plt.plot(simulated_ARMA_1_1_points)

plt.title("Simulated ARMA(1,1) Process")

plt.xlim([0, 200])

plt.show()
plot_acf(simulated_ARMA_1_1_points);

plot_pacf(simulated_ARMA_1_1_points);
ar2 = np.array([1, 0.6, 0.4])

ma2 = np.array([1, -0.2, -0.5])



simulated_ARMA_2_2_points = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)
plt.figure(figsize=[15, 7.5]); # Set dimensions for figure

plt.plot(simulated_ARMA_2_2_points)

plt.title("Simulated ARMA(2,2) Process")

plt.xlim([0, 200])

plt.show()
plot_acf(simulated_ARMA_2_2_points);

plot_pacf(simulated_ARMA_2_2_points);
np.random.seed(200)



ar_params = np.array([1, -0.4])

ma_params = np.array([1, -0.8])



returns = ArmaProcess(ar_params, ma_params).generate_sample(nsample=1000)



returns = pd.Series(returns)

drift = 100



price = pd.Series(np.cumsum(returns)) + drift
returns.plot(figsize=(15,6), color=sns.xkcd_rgb["orange"], title="simulated return series")

plt.show()
price.plot(figsize=(15,6), color=sns.xkcd_rgb["baby blue"], title="simulated price series")

plt.show()
log_return = np.log(price) - np.log(price.shift(1))

log_return = log_return[1:]
_ = plot_acf(log_return,lags=10, title='log return autocorrelation')
_ = plot_pacf(log_return, lags=10, title='log return Partial Autocorrelation', color=sns.xkcd_rgb["crimson"])
from statsmodels.tsa.arima_model import ARIMA



def fit_arima(log_returns):

        ar_lag_p = 1

        ma_lag_q = 1

        degree_of_differentiation_d = 0



        # create tuple : (p, d, q)

        order = (ar_lag_p, degree_of_differentiation_d, ma_lag_q)



        # create an ARIMA model object, passing in the values of the lret pandas series,

        # and the tuple containing the (p,d,q) order arguments

        arima_model = ARIMA(log_returns.values, order=order)

        arima_result = arima_model.fit()



        #TODO: from the result of calling ARIMA.fit(),

        # save and return the fitted values, autoregression parameters, and moving average parameters

        fittedvalues = arima_result.fittedvalues

        arparams = arima_result.arparams

        maparams = arima_result.maparams



        return fittedvalues,arparams,maparams

fittedvalues,arparams,maparams = fit_arima(log_return)

arima_pred = pd.Series(fittedvalues)

plt.plot(log_return, color=sns.xkcd_rgb["pale purple"])

plt.plot(arima_pred, color=sns.xkcd_rgb["jade green"])

plt.title('Log Returns and predictions using an ARIMA(p=1,d=1,q=1) model');

print(f"fitted AR parameter {arparams[0]:.2f}, MA parameter {maparams[0]:.2f}")