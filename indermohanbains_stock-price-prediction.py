# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fundamentals = '../input/nyse/fundamentals.csv'
prices_split_adjusted = '../input/nyse/prices-split-adjusted.csv'
prices = '../input/nyse/prices.csv'
securities = '../input/nyse/securities.csv'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
fundamentals = pd.read_csv (fundamentals)
prices_split_adjusted = pd.read_csv (prices_split_adjusted)
prices = pd.read_csv (prices)
securities = pd.read_csv (securities)
fundamentals.head ()
prices_split_adjusted.head ()
prices.head ()
securities.head ()
prices_subset = prices.loc [(prices ['symbol'] == 'EBAY') | (prices ['symbol'] == 'NVDA') | (prices ['symbol'] == 'YHOO') | (prices ['symbol'] == 'AAPL'), :]

prices_subset.tail ()
price_table = pd.pivot_table (data = prices_subset, index = 'date', columns = 'symbol', values = 'close' )
df = pd.DataFrame (price_table, columns = ['AAPL', 'EBAY', 'NVDA', 'YHOO'])
df.index = pd.to_datetime (df.index)
df.head ()
round (df [['AAPL', 'EBAY', 'NVDA', 'YHOO']].describe (),2)
sns.heatmap (df.corr (), annot = True)
plt.show ()
fig = plt.figure (figsize = (10,7))
ax1 = plt.subplot (2,1,1)
df [['EBAY', 'NVDA', 'YHOO']].plot (ax = ax1)

ax2 = plt.subplot (2,1,2)
df [['AAPL']].plot (ax = ax2)

plt.tight_layout ()
plt.show ()
fig = plt.figure (figsize = (10,7))
df.boxplot ()
plt.show ()
df.shape
test = df.iloc [(df.index >= '2016-12-01') & (df.index < '2016-12-30'), :]
train = df.iloc [df.index < '2016-12-01', :]

X_train = train [['EBAY', 'NVDA', 'YHOO']]
y_train = train ['AAPL']

X_test = test [['EBAY', 'NVDA', 'YHOO']]
y_test = test ['AAPL']
#base model
from math import sqrt
# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = sqrt (mean_squared_error (y_test, predictions))
print(round (score))
df1 = pd.concat ([X_train, y_train], axis =1)
print (df1.shape)
print ('\n')
print (X_test.shape, '\t', y_test.shape)
#transforming function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

df_perc = df1.rolling(50).apply (percent_change).dropna ()
df_perc.loc["2014":"2015"].plot(figsize = (10,5))
plt.xticks (rotation = 90)
plt.show()
df_perc.shape
def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))
    
    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std (series) * 3)
    
    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian (series)
    return series

# Apply your preprocessing function to the timeseries and plot the results
df_perc = df_perc.apply (replace_outliers)
df_perc.loc["2014":"2015"].plot(figsize = (10,5))
plt.xticks (rotation = 90)
plt.show()
# Define a rolling window with Pandas, excluding the right-most datapoint of the window
df_perc_rolling = df_perc.rolling(50, min_periods=5, closed = 'right')

# Define the features you'll calculate for each window
features_to_calculate = [np.min, np.max, np.mean, np.std]
features.columns = ['AAPL_min', 'AAPL_max', 'AAPL_mean', 'AAPL_std', 'EBAY_min', 'EBAY_max', 'EBAY_mean', 'EBAY_std','NVDA_min', 'NVDA_max', 'NVDA_mean', 'NVDA_std', 'YHOO_min', 'YHOO_max', 'YHOO_mean', 'YHOO_std' ]
# Calculate these features for your rolling window object
features = df_perc_rolling.aggregate(features_to_calculate)
features = features.apply (lambda x : x.fillna (np.nanmedian (x)))

# Plot the results

features.loc[:"2011-01"].boxplot(figsize = (15,5))
plt.xticks (rotation = 90)
plt.show()
features.shape
# percentile features
from functools import partial
percentiles = [1, 10, 25, 50, 75, 90, 99]

# Use a list comprehension to create a partial function for each quantile
percentile_functions = [partial (np.percentile, q=percentile) for percentile in percentiles]
# Calculate each of these quantiles on the data using a rolling window
df_perc_rolling = df_perc.rolling(50, min_periods=5, closed = 'right')
features_percentiles = df_perc_rolling.agg (percentile_functions).apply (lambda x : x.fillna (np.nanmedian (x)))
features_percentiles.columns = ['AAPL_perc_1','AAPL_perc_10','AAPL_perc_25','AAPL_perc_50','AAPL_perc_75','AAPL_perc_90','AAPL_perc_99', 'EBAY_perc_1','EBAY_perc_10','EBAY_perc_25','EBAY_perc_50','EBAY_perc_75','EBAY_perc_90','EBAY_perc_99', 'NVDA_perc_1','NVDA_perc_10','NVDA_perc_25','NVDA_perc_50','NVDA_perc_75','NVDA_perc_90','NVDA_perc_99', 'YHOO_perc_1','YHOO_perc_10','YHOO_perc_25','YHOO_perc_50','YHOO_perc_75','YHOO_perc_90','YHOO_perc_99']

features_percentiles.loc[:'2011-01-01'].boxplot(figsize = (15,5))

plt.xticks (rotation = 90)
plt.show()
features_percentiles.shape
# These are the "time lags"
shifts = np.arange(1,3).astype(int)

df_shifted = pd.DataFrame (index = df_perc.index, columns = None)
df_shifted.fillna (0, inplace = True)

for day_shift in shifts:
    for x in df_perc.columns:
   
        # Use a dictionary comprehension to create name: value pairs, one pair per shift
        shifted_data = {"lag_{}_day_{}".format(day_shift,x): df_perc [x].shift(day_shift) for day_shift in shifts}
    
        # Convert into a DataFrame for subsequent use
        shifted = pd.DataFrame (shifted_data)
    
                   
        df_shifted = pd.concat ([df_shifted, shifted], axis = 1)
        df_shifted = df_shifted.apply (lambda w : w.fillna (np.nanmedian (w)))
# Plot the first 100 samples of each
df_shifted.boxplot(figsize = (25,5))
plt.xticks (rotation = 90)
plt.show()
df_shifted.shape
X_tot = pd.concat ([df1,features,features_percentiles, df_shifted ], axis = 1).dropna ()
# date features
X_tot['day_of_week'] = X_tot.index.dayofweek
X_tot['week_of_year'] = X_tot.index.weekofyear
X_tot['month_of_year'] = X_tot.index.month
# date features
X_test['day_of_week'] = X_test.index.dayofweek
X_test['week_of_year'] = X_test.index.weekofyear
X_test['month_of_year'] = X_test.index.month
Y = X_tot ['AAPL']
X_tot = X_tot.drop ('AAPL', axis = 1)
X_test = X_test.merge (X_tot.drop (['EBAY', 'NVDA', 'YHOO'], axis = 1), on = ['day_of_week', 'week_of_year', 'month_of_year'], how = 'left').drop_duplicates (subset = ['day_of_week', 'week_of_year', 'month_of_year'], keep = 'last')
X_tot.shape
X_test.shape
#model
from sklearn.metrics import mean_squared_error
# Fit our model and generate predictions
model = Ridge(11000)
model.fit(X_tot, Y)
predictions = model.predict(X_test)
score = sqrt (mean_squared_error (y_test, predictions))
print(round (score))
#feature importance
plt.figure (figsize = (20,4))
coefs = pd.Series (dict (zip (list (X_tot.columns), list(abs (model.coef_)))))
abs (coefs).sort_values (ascending = False).plot.bar ()
plt.show ()
plt.scatter (x = y_test, y = predictions)
Accuracy = round (np.corrcoef (y_test, predictions)[0,1]*100)

print ('The apple stock price for Dec, 2016 has been predicted with accuracy of {}'.format (Accuracy))
