# include the libraries needed
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
# set figure size globally
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = (15,6)

# Turn off warnings to prepare the report
import warnings
warnings.filterwarnings('ignore')
timeseries = pd.read_csv('../input/IPG2211A2N.csv', index_col=0, parse_dates=True)
timeseries.head()
# adjust the index to take account that the data is monthly data
timeseries = timeseries.asfreq('M', method='pad', how='start')

# use past 11 years instead of using the hole dataset
timeseries = timeseries['2008-01-01':]

# plot the time series
timeseries.plot(title='Industrial Production Index 2008-2018');
# calculate rolling statistics for 1 year window
rollavg_1year = timeseries.rolling(window=12).mean()
rollstd_1year = timeseries.rolling(window=12).std()

plt.plot(timeseries, color='black', label='Original Data', linewidth=1)
plt.plot(rollavg_1year, color='red', label='Mean', linewidth=1)
plt.plot(rollstd_1year, color='blue', label='STD', linewidth=1)
plt.legend()
plt.title('Rolling Mean & STD with window size of 12');
# apply the ADF test
test_result = adfuller(timeseries['IPG2211A2N'])

# see if we can reject null-hypothesis
dfoutput = pd.Series(test_result[0:2], index=['Test Statistic','p-value'])
print(dfoutput)
if dfoutput[1] <= 0.05:
    print('-> Reject Null Hypothesis. Data is stationary')
else:
    print("-> can't reject Null Hypothesis. Data is non-stationary")
# use the differencing method with a time shift of 1
timeseries_diff = timeseries.diff()

# remove the  NAN values resulting from the differencing
timeseries_diff.dropna(inplace=True)
timeseries_diff.head(10)
timeseries_diff_avg = timeseries_diff.rolling(12).mean()
timeseries_diff_std = timeseries_diff.rolling(12).std()

plt.plot(timeseries_diff, color='black', label='Original Data', linewidth=0.1)
plt.plot(timeseries_diff_avg, color='red', label='Mean', linewidth=1)
plt.plot(timeseries_diff_std, color='blue', label='STD', linewidth=1)
plt.legend()
plt.title('Rolling Statistics after Differencing');

# carry out stationary test
test_result = adfuller(timeseries_diff['IPG2211A2N'])

# see if we can reject null-hypothesis
dfoutput = pd.Series(test_result[0:2], index=['Test Statistic','p-value'])
print(dfoutput)
if dfoutput[1] <= 0.05:
    print('-> Reject Null Hypothesis. Data is stationary')
else:
    print("-> can't reject Null Hypothesis. Data is non-stationary")
# calculate the autocorrelation
calculated_acf = acf(timeseries_diff, nlags=12)

# calculate the partial autocorrelation
calculated_pacf = pacf(timeseries_diff, nlags=12)

upper_bound_confidence_level = 1.96/np.sqrt(len(timeseries_diff))
lower_bound_confidence_level = -1.96/np.sqrt(len(timeseries_diff))

#Plot ACF:    
plt.subplot(121)    
plt.plot(calculated_acf)
plt.axhline(y = 0, linestyle='--')
plt.axhline(y = upper_bound_confidence_level, linestyle='--')
plt.axhline(y = lower_bound_confidence_level, linestyle='--')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(calculated_pacf)
plt.axhline(y=upper_bound_confidence_level, linestyle='--')
plt.axhline(y=0, linestyle='--')
plt.axhline(y=lower_bound_confidence_level, linestyle='--')
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
# create a list of results for SARIMA model
# with all parameters to choose the best model later on
aic_list = []
# set the max range of search to 3 for each parameter
p_range = d_range = q_range = range(0, 3)
for p in p_range:
    for d in d_range:
        for q in q_range:
            for P in p_range:
                for D in d_range:
                    for Q in q_range:
                        model = SARIMAX(timeseries,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False,
                                        seasonal_order=(P,D,Q,12),
                                        order=(p, d, q),
                                        freq='M')
                        # fit the model
                        ret = model.fit()
                        
                        # store the result of this model
                        aic_list.append((ret.aic, p, d, q, P, D, Q))
sorted_list = sorted(aic_list, key=lambda x: x[0])

print('BEST ARIMA(%d,%d,%d)(%d,%d,%d)12 AIC=%f' %
      (sorted_list[0][1], sorted_list[0][2], sorted_list[0][3],
      sorted_list[0][4],sorted_list[0][5],sorted_list[0][6], sorted_list[0][0]))
# prepare the training data
training_data = timeseries

# create and train the model
model = SARIMAX(training_data,
                enforce_stationarity=False,
                enforce_invertibility=False,
                seasonal_order=(0,2,2,12),
                order=(0, 1, 2),
                freq='M')

# fit the model
ret = model.fit()

# print summary of the model
print(ret.summary())
plt.plot(timeseries['2012':], color='black', label='Original Data', linewidth=1)
plt.plot(ret.fittedvalues['2012':], color='red', label='Fitted Data', linewidth=1)
plt.legend()
plt.title('Fitted Data vs Original');
ret.plot_diagnostics();
prediction = ret.get_prediction(start='2012-12-31',
                                end='2022-12-31',
                                dynamic=False)
fig = training_data.plot();
prediction.predicted_mean.plot(ax=fig)

prediction_confidence_interval = prediction.conf_int()
fig.fill_between(prediction_confidence_interval.index,
                 prediction_confidence_interval.iloc[:,0],
                 prediction_confidence_interval.iloc[:,1], alpha=0.1);