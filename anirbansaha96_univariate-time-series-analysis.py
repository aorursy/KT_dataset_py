from statsmodels.graphics.tsaplots import plot_pacf                 #Used to Plot PACF plot.

from statsmodels.graphics.tsaplots import plot_acf                  #Used to Plot ACF plot.

from statsmodels.tsa.arima_process import ArmaProcess               #Used to generate data point

from statsmodels.tsa.stattools import pacf                          #Used to find PACF characteristics.

from statsmodels.regression.linear_model import yule_walker         #Used to estimate parameters 

from statsmodels.tsa.stattools import adfuller                      #Used to check stationarity

import matplotlib.pyplot as plt               

import numpy as np

%matplotlib inline
ar3 = np.array([1, 0.8, 0.6,0.1])

ma = np.array([1])

simulated_AR3_data = ArmaProcess(ar3, ma).generate_sample(nsample=1000)

plt.figure(figsize=[10, 7.5]); # Set dimensions for figure

plt.plot(simulated_AR3_data)

plt.title("Simulated AR(3) Process")

plt.show()
plot_acf(simulated_AR3_data,lags=20);
plot_pacf(simulated_AR3_data,lags=20);
rho, sigma = yule_walker(simulated_AR3_data, 3, method='mle')

print(f'rho: {-rho}')

print(f'sigma: {sigma}')
ar3 = np.array([1])

ma3 = np.array([1, 0.9, 0.3,0.8])

MA3_process = ArmaProcess(ar3, ma3).generate_sample(nsample=1000)
plt.figure(figsize=[10, 7.5]); # Set dimensions for figure

plt.plot(MA3_process)

plt.title('Moving Average Process of Order 3')

plt.show()
plot_acf(MA3_process, lags=20)

plt.show()
plot_pacf(MA3_process,lags=20)

plt.show()
arparams = [.75, -.25]

maparams = [.65, .35]

arma_process = ArmaProcess.from_coeffs(arparams, maparams).generate_sample(nsample=1000)
plt.figure(figsize=[10, 7.5]); # Set dimensions for figure

plt.plot(arma_process)

plt.title('ARMA Process of Order (2,2)')

plt.show()
plot_acf(arma_process, lags=20)

plt.show()
plot_pacf(arma_process,lags=20)

plt.show()
import pandas as pd

url = 'https://github.com/marcopeix/time-series-analysis/blob/master/data/jj.csv?raw=True'

df = pd.read_csv(url)

print(df.head(5))
import matplotlib.pyplot as plt

plt.figure(figsize=[15, 7.5]); # Set dimensions for figure

plt.scatter(df['date'], df['data'])

plt.title('Quarterly EPS for Johnson & Johnson')

plt.ylabel('EPS per share ($)')

plt.xlabel('Date')

plt.xticks(rotation=90)

plt.grid(True)

plt.show()
from statsmodels.graphics.tsaplots import plot_pacf                 #Used to Plot PACF plot.

from statsmodels.graphics.tsaplots import plot_acf                  #Used to Plot ACF plot.

plot_acf(df['data'],lags=20)

plot_pacf(df['data'],lags=20)

plt.show()
from statsmodels.tsa.stattools import adfuller

ad_fuller_result = adfuller(df['data'])

print(f'ADF Statistic: {ad_fuller_result[0]}')

print(f'p-value: {ad_fuller_result[1]}')
import numpy as np

df['data'] = np.log(df['data'])

df['data'] = df['data'].diff()

df = df.drop(df.index[0])

df.head()
plt.figure(figsize=[15, 7.5]); # Set dimensions for figure

plt.plot(df['data'])

plt.title("Log Difference of Quarterly EPS for Johnson & Johnson")

plt.show()
ad_fuller_result = adfuller(df['data'])

print(f'ADF Statistic: {ad_fuller_result[0]}')

print(f'p-value: {ad_fuller_result[1]}')
plot_pacf(df['data'], lags=20);

plot_acf(df['data'], lags=20);

plt.show()
from itertools import product 

from tqdm import tqdm_notebook

ps = range(0, 8, 1)

d = 1

qs = range(0, 8, 1)

# Create a list with all possible combination of parameters

parameters = product(ps, qs)

parameters_list = list(parameters)

order_list = []

for each in parameters_list:

    each = list(each)

    each.insert(1, 1)

    each = tuple(each)

    order_list.append(each)
%%capture

import statsmodels.api as sm



results=[]

for order in order_list:

    try: 

        model = sm.tsa.statespace.SARIMAX(df['data'], order=order).fit(disp=-1);

    except:

        continue  

    aic = model.aic;

    results.append([order, model.aic]);
result_df = pd.DataFrame(results)

result_df.columns = ['(p, d, q)', 'AIC']

#Sort in ascending order, lower AIC is better

result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
result_df.head(5)
best_model = sm.tsa.statespace.SARIMAX(df['data'], order=(3,1,3)).fit()

print(best_model.summary())
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung_box, p_value = acorr_ljungbox(best_model.resid)

print(f'Ljung-Box test: {ljung_box[:10]}')

print(f'p-value: {p_value[:10]}')
plot_pacf(best_model.resid,lags=20);

plot_acf(best_model.resid,lags=20);

plt.show()