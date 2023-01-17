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
df = pd.read_csv("/kaggle/input/others/daily-total-female-births-CA.csv")
df.head()
type(df)
df.info()
df = pd.read_csv("/kaggle/input/others/daily-total-female-births-CA.csv", parse_dates=[0])
df.head()
df.info()
# Index is our time series date data and vales are births columns.

series = pd.read_csv("/kaggle/input/others/daily-total-female-births-CA.csv", parse_dates=[0], index_col=[0], squeeze=True)
series.head()
print(series.shape)
type(series)
series
print(series['1959-05'])
series.describe()
import matplotlib.pyplot as plt
df['births'].plot()
df.index = df.date # X-axis is date column now
print(df.head())
df['births'].plot()
dfplot = df[(df['date']>'1959-03-01') & (df['date']<'1959-06-01')]
dfplot.births.plot()
import seaborn as sns
df = pd.read_csv("/kaggle/input/others/daily-total-female-births-CA.csv", parse_dates=[0])
sns.regplot(x=df.index.values, y=df.births)

### Increasing Trend
### Quadratic TL
sns.regplot(x=df.index.values, y=df.births, order=2)
### Cubic TL
sns.regplot(x=df.index.values, y=df.births, order=3)
us_miles = pd.read_csv('../input/others/us-airlines-monthly-aircraft-miles-flown.csv', parse_dates=[0])
us_miles.head()
us_miles['MilesMM'].plot() # This shows seasonality, peak during end of the year
sns.regplot(x=us_miles.index.values, y=us_miles['MilesMM'])
us_miles['year'] = us_miles['Month'].dt.year
us_miles.head()
us_miles.groupby('year')['year','MilesMM'].head(5)
print(us_miles.groupby('year')['MilesMM'].mean())
us_miles.groupby('year')['MilesMM'].mean().plot()
plt.title('Seasonality Removed - MilesMM')
us_miles['lags']= us_miles['MilesMM'].shift(1)
us_miles.head()
sns.scatterplot(x=us_miles['lags'], y=us_miles['MilesMM'])
plt.title('Positive Correlation is exhibited')
# Also you can use
from pandas.plotting import lag_plot
lag_plot(us_miles['MilesMM'])
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(us_miles['MilesMM'])

# X-axis is lag value +1 (like previous plt) , +2, +3, ...90 lags and \
# Y-axis is correlation of actual MilesMM and it's lagged values, +ve means positive correlation and -ve vice versa.

# From the plot we see the first 5 lags are highly correlated with actual MilesMM feature.
df = pd.read_csv('../input/others/daily-total-female-births-CA.csv', parse_dates=[0])
df.head()
features = df.copy()
features['Year'] = df['date'].dt.year
features['Month'] = df['date'].dt.month
features['Days'] = df['date'].dt.day

# New features (year, month, days from date)
features.head()
features['lag1'] = df['births'].shift(1)
features['lag2'] = df['births'].shift(365)

features.head()
# Take n and n-1 values of all date values and averages them into a single unit (window = 2)
# Take n , n-1, and n-2 values of all date values and averages them into a single unit (window = 3)

features['Roll_mean'] = df['births'].rolling(window=2).mean() 
features['Roll_Max'] = df['births'].rolling(window=3).max() # Mean or Max or Min

features.head(10)
features['Expanding_Max'] = df['births'].expanding().max()
features.head(10)
df = pd.read_csv('../input/others/us-airlines-monthly-aircraft-miles-flown.csv', parse_dates=[0])
df.head()
# Downsampling (12 months = 4 Quarters)
# 'Q' - Quaterly, 'A' - Annually
quaterly_miles = df.resample('Q', on='Month').mean() # Quarter has 3 sets and taking mean out of it
quaterly_miles.head()
yearly_miles = df.resample('A', on='Month').sum()
yearly_miles.head()
daily_miles = df.resample('D', on='Month').mean() # Only creates structure, later fill them
daily_miles.head(50)
# To fill the data between day 1 and day 30, we can interpolate a linear function to fill those values
interpolated_df = daily_miles.interpolate(method='linear')
interpolated_df.head(50)
interpolated_df.plot()
# For smoothing , we replace linear with polynomial function (quadratic= order=2, third degree polynomial order=4)
poly_interpolated_df = daily_miles.interpolate(method='spline', order=2)
poly_interpolated_df.head(50)
poly_interpolated_df.plot()
# Comparing the two plots for smoothening
interpolated_df.plot()
poly_interpolated_df.plot()
from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_csv('../input/others/us-airlines-monthly-aircraft-miles-flown.csv', parse_dates=[0])
df.index = df.Month
df.head()
results_add = seasonal_decompose(df['MilesMM'], model='additive')
results_add.plot()
# Original, Trend, Seasonality, Noise (Residual vales)
results_mul = seasonal_decompose(df['MilesMM'], model='multiplicative')
results_mul.plot()
# Original, Trend, Seasonality, Noise (Residual vales)
df['lags'] = df['MilesMM'].shift(1)
df.head()
df['MilesMM-lags'] = df['MilesMM'].diff(periods=1)
df.head()
# Check for those 3 patterns in the original dataset
df.index = df.Month
result_1 = seasonal_decompose(df['MilesMM'], model='additive')
result_1.plot()
# The differencing must have removed the trend, but not the seasonality. let's check
# Refer the y-axis, the range is less which means there's no trend
result_2 = seasonal_decompose(df.iloc[1:,3], model='additive')
result_2.plot()
df.head()
df['MilesMM'].plot()
df['MilesMM-lags'].plot() # Seasonality for first 3 values
df['MilesMM-lags12'] = df['MilesMM'].diff(periods=12)
df['MilesMM-lags12'].plot()
df.head(20)
# The second differencing must have removed the seasonality. let's check
# Refer the y-axis, the range is less which means there's no seasonality and trend
result_3 = seasonal_decompose(df.iloc[12:,4], model='additive')
result_3.plot()
