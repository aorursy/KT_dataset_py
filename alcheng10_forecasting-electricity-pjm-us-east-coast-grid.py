import pandas as pd
import numpy as np
from datetime import datetime

# Configure matplotlib plotting
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use('fivethirtyeight')

from pylab import rcParams
rcParams['figure.figsize'] = 11, 9 #Changes default matplotlib plots to this size
# Load my EDA helper function created to do some high-level analysis
class EDA():

    df = pd.DataFrame()
    
    def __init__(self, df):
        '''
        Creates EDA object for the DataFrame
        
        Note for time series data, have the index be the timestamp prior to creating this Object.
        
        :param df : DataFrame
        '''
        self.df = df
        
    def missing_values(self):
        '''
        Checks missing values
        
        :return DataFrame
        
        '''
        missing = self.df[self.df.isna().any(axis=1)]
        
        print("Missing values data")
        
        return missing
    
    def duplicate_values(self):
        duplicates = self.df[self.df.duplicated(subset=None, keep='first')==True]
        
        print("Duplicate values data")
        
        return duplicates
        
    def duplicate_indices(self):
        '''
        Check whether the indices have any duplicates
        
        :return DataFrame
        '''        
        duplicate_indices = self.df[self.df.index.duplicated()==True]
        
        print("Duplicate indices")
        
        return duplicate_indices
            
    def summary(self):
        '''
        Return summary/describe of DataFrame
        
        :return DataFrame
        '''
        df = self.df.reset_index() # Reset to include the index
        
        summary = df.describe(include='all').transpose()
        
        print("Summary metrics")
        
        return summary
    
    def pandas_profiling(self):
        import pandas_profiling
        
        self.df.profile_report(style={'full_width':True})  
    
    def histogram_KDE(self):
        ''' 
        :return seaborn plot
        '''       
        sns.pairplot(self.df, diag='kde')
        sns.distplot(kde=True, fit=[st.norm or st.lognorm])
        
    def outliers(self, col):
        ''' 
        Checks outliers - anything outside of 5% to 95% quartile range
        
        :param col : str
            Name of col to be tested
            
        :return DataFrame
        '''
        outliers = self.df[~self.df[col].between(self.df[col].quantile(.05), self.df[col].quantile(.95))]
        
        print("Outliers")
        
        return outliers
        
    def missing_timeseries_points(self, freq='D'):
        '''
        Checks whether there's any missing data points in continuous time series data.
        
        :param freq optional default = 'D' : str
            Frequency compliant with pandas formatting
        
        :return DataFrame
        '''
        # First create date range
        date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)

        # Now compare against dataset
        missing_timeseries = self.df.index[~self.df.index.isin(date_range)]
        
        print("Missing timeseries data")
        
        return missing_timeseries

    def corr_heatmap(df):
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = self.df.corr()
        hm = sns.heatmap(round(corr,2), annot=True, cmap="coolwarm",fmt='.2f', linewidths=.05)
        fig.subplots_adjust(top=0.93)
        title = fig.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)

        plt.show()

    def plot_time_series_seasonal_decomp(self, type='add'):
        '''
        Plots seasonal decomposition of timeseries data
        
        :return matplotlib Plot
        '''
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(self.df, model='multiplicative')

        fig = decomposition.plot()
        plt.show()
   
    def time_series_ADF(self):            
        '''
        Returns Augmented Dickey-Fuller Test
        '''
        from statsmodels.tsa.stattools import adfuller as ADF

        series = data['KwH'] # ADF takes series, not DF

        result = ADF(series)

        print('ADF Statistic: %f4.2' % result[0])
        print('P-value %f4.2' % result[1])
# Load the Data
data = pd.read_csv("../input/hourly-energy-consumption/PJME_hourly.csv")

data.set_index('Datetime',inplace=True)

data.index = pd.to_datetime(data.index)

data_copy = data.copy(deep=True) # Make a deep copy, including a copy of the data and the indices
data.head(10)
eda_helper = EDA(data)

eda_helper.summary()
eda_helper.missing_values()
eda_helper.duplicate_values()
eda_helper.duplicate_indices()
def preprocess_data(df):
    # We have duplicates, so we need to de-duplicate it - grabbing first value that pops up for a time series
    df.sort_index()
    df = df.groupby(df.index).first()
    
    #Set freq to H
    df = df.asfreq('H')
    
    return df
data = preprocess_data(data)

data.index
# Let's visualise the data
data.plot()

plt.show()
# Given there's no missing data, we can resample the data to daily level
daily_data = data.resample(rule='D').sum()

# Set frequency explicitly to D
daily_data = daily_data.asfreq('D')

daily_data.head(10)
# We can confirm it is at the right frequency
daily_data.index
daily_data.plot()

plt.show()
daily_data = daily_data.drop([daily_data.index.min(), daily_data.index.max()])
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(daily_data, model='additive')

fig = decomposition.plot()
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose
weekly_data = data.resample(rule='W').sum()
decomposition = seasonal_decompose(weekly_data, model='additive') # Aggregate to weekly level

fig = decomposition.plot()
plt.show()
# Create new dataset for heatmap
heatmap_data = daily_data.copy()

# First we need to add weekdays as a column
heatmap_data['Weekday_Name'] = daily_data.index.weekday_name

# Next we add the year as column and group the data up to annual day of week level
heatmap_data['Year'] =  heatmap_data.index.year
heatmap_data = heatmap_data.groupby(['Year', 'Weekday_Name']).sum()

# Reset index 
heatmap_data = heatmap_data.reset_index()

# We drop off 2018 because it's not a full year
heatmap_data = heatmap_data[heatmap_data['Year'] != 2018]

# Pivot it to a uniform data format for heatmaps
heatmap_data = heatmap_data.pivot(index='Year', columns='Weekday_Name', values='PJME_MW')

# Reorder columns
heatmap_data = heatmap_data[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

heatmap_data.head(100)
# Visualise electricity load via Heatmap
sns.heatmap(heatmap_data, linewidths=.5, cmap='YlOrRd', cbar=True, cbar_kws={"format": '%1.0f MWh'}).set_title('Heatmap - by Day of Week')
# Create new dataset for heatmap
heatmap_data = data.copy()

# First we need to add weekdays as a column
heatmap_data['Hour'] = data.index.hour

# Next we add the year as column and group the data up to annual day of week level
heatmap_data['Year'] =  heatmap_data.index.year
heatmap_data = heatmap_data.groupby(['Year', 'Hour']).sum()

# Reset index 
heatmap_data = heatmap_data.reset_index()

# We drop off 2018 because it's not a full year
heatmap_data = heatmap_data[heatmap_data['Year'] != 2018]

# Pivot it to a uniform data format for heatmaps
heatmap_data = heatmap_data.pivot(index='Year', columns='Hour', values='PJME_MW')

heatmap_data.head(100)
# Visualise electricity load via Heatmap
sns.heatmap(heatmap_data, linewidths=.5, cmap='YlOrRd', cbar=True, cbar_kws={"format": '%1.0f MWh'}).set_title('Heatmap - by Hour of Day')
# Create new dataset for heatmap
heatmap_data = daily_data.copy()

# First we need to add weekdays as a column
heatmap_data['Month'] = daily_data.index.month_name()

# Next we add the year as column and group the data up to annual day of week level
heatmap_data['Year'] =  heatmap_data.index.year
heatmap_data = heatmap_data.groupby(['Year', 'Month']).sum()

# Reset index
heatmap_data = heatmap_data.reset_index()

# We drop off 2018 because it's not a full year
heatmap_data = heatmap_data[heatmap_data['Year'] != 2018]

# Pivot it to a uniform data format for heatmaps
heatmap_data = heatmap_data.pivot(index='Year', columns='Month', values='PJME_MW')

# Reorder columns
heatmap_data = heatmap_data[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']]

heatmap_data.head(10)
# Visualise electricity load via Heatmap
sns.heatmap(heatmap_data, linewidths=.5, cmap='YlOrRd', cbar=True, cbar_kws={"format": '%1.0f MWh'}).set_title('Heatmap - by Day of Week')
# First let's load the data
weather_data_2017 = pd.read_csv("../input/ncei-climate-data-washington-dc-temperature/Washington_DC_Weather_Avg_Temp_2016-17.csv")
weather_data_2018 = pd.read_csv("../input/ncei-climate-data-washington-dc-temperature/Washington_DC_Weather_Avg_Temp_2018.csv")

weather_data = weather_data_2017.append(weather_data_2018)

weather_data.head(10)
eda_helper = EDA(weather_data)

eda_helper.summary()
# Clean data and aggregate
weather_data = weather_data[['DATE', 'TAVG']]
weather_data = weather_data[~weather_data['TAVG'].isna()]

weather_data.groupby(['DATE']).mean()
weather_data = weather_data.set_index('DATE')
weather_data.index = pd.to_datetime(weather_data.index)

# Sort to make sure plotting works
weather_data = weather_data.sort_values(by='DATE', ascending=True)
plt.plot(weather_data, label='Average Temp (°C)', color='lightseagreen')

# Plot Labels, Legends etc
plt.title('Washington D.C. Average Temperature')
plt.legend(loc='best')
plt.xlabel("Timestamp")
plt.ylabel("Degrees (°C)")
plt.legend(loc='best')

plt.show()
fig, ax = plt.subplots(2,1, figsize=(20,20))

# Plot 1
ax[0].plot(weather_data['2016-01-01':'2018-08-02'], label='Average Temp (°C)', color='lightseagreen')
ax[0].set_title('Washington D.C. Average Temperature')
ax[0].set_ylabel("Degrees (°C)")
ax[0].legend(loc='best')

# Plot 2
ax[1].plot(daily_data['2016-01-01':'2018-08-02'], label='PJME MW Daily Load (MW)', color='royalblue')
ax[1].set_title('PJM East Region Electricity Load')
ax[1].set_xlabel('Timestamp')
ax[1].set_ylabel("Load (MW)")
ax[1].legend(loc='best')

# Add vertical lines to emphasis point
import datetime as dt
ax[0].axvline(dt.datetime(2016, 8, 15), color='red', linestyle='--')
ax[1].axvline(dt.datetime(2016, 8, 15), color='red', linestyle='--')
ax[0].axvline(dt.datetime(2016, 12, 15), color='red', linestyle='--')
ax[1].axvline(dt.datetime(2016, 12, 15), color='red', linestyle='--')
ax[0].axvline(dt.datetime(2017, 7, 15), color='red', linestyle='--')
ax[1].axvline(dt.datetime(2017, 7, 15), color='red', linestyle='--')
ax[0].axvline(dt.datetime(2018, 1, 1), color='red', linestyle='--')
ax[1].axvline(dt.datetime(2018, 1, 1), color='red', linestyle='--')

plt.show()
correlation = daily_data['2016-01-01':'2018-08-02']['PJME_MW'].corr(weather_data['2016-01-01':'2018-08-02']['TAVG'], method='pearson')

print("The correlation between the PJM East Region electricity load and Washington D.C. average temperature is: {}%".format(correlation*100))
from statsmodels.tsa.stattools import adfuller as ADF

series = daily_data['PJME_MW'] # ADF takes series, not DF

result = ADF(series)

print('ADF Statistic: ', result[0])
print('P-value: {:.20f}'.format(result[1]))
from statsmodels.stats.diagnostic import het_breuschpagan as BP
import statsmodels.api as sm
from statsmodels.formula.api import ols

bp_data = daily_data.copy()
bp_data['Time_Period'] = range(1, len(bp_data)+1) # Convert time series points into consecutive ints

formula = 'PJME_MW ~ Time_Period' # ie PJME MW depends on Time Period (OLS auto adds Y intercept)

# Next we apply Ordinary Linear Square baseline regression model - as baseline test
model = ols(formula, bp_data).fit()

result = BP(model.resid, model.model.exog)

print('ADF Statistic: ', result[0])
print('P-value: {:.20f}'.format(result[1]))
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

#fig, ax = plt.subplots(2,1)

# Plot the acf function
plot_acf(daily_data['PJME_MW'],lags=720) #alpha 1 suppresses CI

plt.show()
plot_pacf(daily_data['PJME_MW'],lags=720) #alpha 1 suppress CI

plt.show()
# First we split it up between train and test
# We will aim for a 12 month forecast horizon (ie predict the last 12 months in the dataset)
cutoff = '2017-08-03'

daily_data.sort_index()

train = daily_data[:cutoff]
test = daily_data[cutoff:]
baseline_prediction = train['2016-08-03':'2017-08-02']

baseline_prediction.index = pd.date_range(start='2017-08-03', end='2018-08-02', freq='D')

baseline_prediction.tail()
# Evaluate it's performance using Mean Absolute Error (MAE)
from statsmodels.tools.eval_measures import meanabs

print("MAE Baseline: {:.20f}".format(meanabs(test['PJME_MW'], baseline_prediction['PJME_MW'])))
# Let's visually see the results
plt.scatter(x=train.index, y=train['PJME_MW'], label='Training Data', color='black')
plt.scatter(x=test.index, y=test['PJME_MW'], label='Test Actuals', color='red')
plt.plot(baseline_prediction, label='Baseline', color='orange')

# Plot Labels, Legends etc
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')
plt.title('Baseline Model (Daily) - Results')

# For clarify, let's limit to only 2017 onwards
plt.xlim(datetime(2017, 1, 1),datetime(2018, 9, 1))

plt.show()
# First construct the residuals - basically the errors
naive_errors = test.copy()
naive_errors['PJME_MW_PREDICTION'] = baseline_prediction['PJME_MW']
naive_errors['error'] = naive_errors['PJME_MW_PREDICTION'] - naive_errors['PJME_MW']

# Let's visually see the errors via scatterplot
plt.scatter(naive_errors.index, naive_errors['error'], label='Residual/Errors')

# Plot Labels, Legends etc
plt.title('Naive One-Year Persistence - Residual/Errors Distribution')
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')

plt.show()
# Plot Histogram with Kernel Density Estimation (KDE)
sns.distplot(naive_errors['error'], kde=True);

# Plot Labels, Legends etc
plt.title('Naive One-Year Persistence - Residual/Errors Distribution')
from statsmodels.graphics.tsaplots import plot_acf

# Plot the acf function
plot_acf(naive_errors['error'],lags=300) #alpha 1 suppresses CI

plt.title('Naive One-Year Persistence - Residual/Errors Autocorrelation')
plt.show()
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# First we split it up between train and test
htrain = train['PJME_MW'] # HWES takes series, not DF
htest = test['PJME_MW'] # HWES takes series, not DF

model = ExponentialSmoothing(
    htrain
    ,trend='add'
    ,seasonal='add'
    ,freq='D'
    ,seasonal_periods=90 #Default is auto estimated - 4 is quarterly and 7 is weekly
).fit(
    optimized=True # Default is True - auto estimates the other parameters using Grid Search
    ,use_basinhopping=True # Uses Basin Hopping Algorithm for optimising parameters
    ,use_boxcox='log' #Boxcox transformation via log
    #,smoothing_level= # Alpha
    #,smoothing_slope= # Beta
    #,smoothing_seasonal= # Gamma
)

HWES_prediction = model.predict(start=htest.index[0], end=htest.index[-1])
HWES_prediction = HWES_prediction.to_frame().rename(columns={0: 'PJME_MW'})

print("Finished training and predicting")

# Let's see what the model did
model.summary()
# Evaluate it's performance using Mean Absolute Error (MAE)
from statsmodels.tools.eval_measures import meanabs

print("MAE HWES ADD: {:.20f}".format(meanabs(htest, HWES_prediction['PJME_MW'])))
# Let's visually see the results
plt.scatter(x=train.index, y=train['PJME_MW'], label='Training Data', color='black')
plt.scatter(x=test.index, y=test['PJME_MW'], label='Test Actuals', color='red')
plt.plot(HWES_prediction['PJME_MW'], label='HWES', color='orange')

# Plot Labels, Legends etc
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')
plt.title('Holtwinters Model (Daily) - Results')

# For clarify, let's limit to only 2017 onwards
plt.xlim(datetime(2017, 1, 1),datetime(2018, 9, 1))

plt.show()
# Feature Engineering first

def preprocess_xgb_data(df, lag_start=1, lag_end=365):
    '''
    Takes data and preprocesses for XGBoost.
    
    :param lag_start default 1 : int
        Lag window start - 1 indicates one-day behind
    :param lag_end default 365 : int
        Lag window start - 365 indicates one-year behind
        
    Returns tuple : (data, target)
    '''
    # Default is add in lag of 365 days of data - ie make the model consider 365 days of prior data
    for i in range(lag_start,lag_end):
        df[f'PJME_MW {i}'] = df.shift(periods=i, freq='D')['PJME_MW']

    df.reset_index(inplace=True)

    # Split out attributes of timestamp - hopefully this lets the algorithm consider seasonality
    df['date_epoch'] = pd.to_numeric(df['Datetime']) # Easier for algorithm to consider consecutive integers, rather than timestamps
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['dayofmonth'] = df['Datetime'].dt.day
    df['dayofyear'] = df['Datetime'].dt.dayofyear
    df['weekofyear'] = df['Datetime'].dt.weekofyear
    df['quarter'] = df['Datetime'].dt.quarter
    df['month'] = df['Datetime'].dt.month
    df['year'] = df['Datetime'].dt.year
    
    x = df.drop(columns=['Datetime', 'PJME_MW']) #Don't need timestamp and target
    y = df['PJME_MW'] # Target prediction is the load
    
    return x, y
example_data = train.copy() #Otherwise it becomes a pointer

example_x, example_y = preprocess_xgb_data(example_data)

example_x.head(10)
xtrain = train.copy() #Otherwise it becomes a pointer
xtest = test.copy() # Otherwise it becomes a pointer

train_feature, train_label = preprocess_xgb_data(xtrain)
test_feature, test_label = preprocess_xgb_data(xtest)
#Train and predict using XGBoost
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split

# We will try with 1000 trees and a maximum depth of each tree to be 5
# Early stop if the model hasn't improved in 100 rounds
model = XGBRegressor(
    max_depth=6 # Default - 6
    ,n_estimators=1000
    ,booster='gbtree'
    ,colsample_bytree=1 # Subsample ratio of columns when constructing each tree - default 1
    ,eta=0.3 # Learning Rate - default 0.3
    ,importance_type='weight' # Default is gain
)
model.fit(
    train_feature
    ,train_label
    ,eval_set=[(train_feature, train_label)]
    ,eval_metric='mae'
    ,verbose=True
    ,early_stopping_rounds=100 # Stop after 100 rounds if it doesn't after 100 times
)

xtest['PJME_MW Prediction'] = model.predict(test_feature)
XGB_prediction = xtest[['Datetime', 'PJME_MW Prediction']].set_index('Datetime')
from sklearn.metrics import mean_absolute_error

print("MAE XGB: {:.20f}".format(mean_absolute_error(test_label, XGB_prediction['PJME_MW Prediction'])))
# Let's visually see the results
plt.scatter(x=train.index, y=train['PJME_MW'], label='Training Data', color='black')
plt.scatter(x=test.index, y=test['PJME_MW'], label='Test Actuals', color='red')
plt.plot(XGB_prediction, label='XGB One-Step Ahead', color='orange')

# Plot Labels, Legends etc
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')
plt.title('XGBoost Model One-Step Ahead (Daily) - Results')

# For clarify, let's limit to only 2015 onwards
plt.xlim(datetime(2015, 1, 1),datetime(2018, 10, 1))


plt.show()
import xgboost as xgb

xgb.plot_importance(model, max_num_features=10, importance_type='weight') # "weight” is the number of times a feature appears in a tree

plt.show()
# So because we need the lag data, we need to preprocess then do the split
all_data = daily_data.copy()

feature, label = preprocess_xgb_data(all_data, lag_start=365, lag_end=720)

# We will aim for a 12 month forecast horizon (ie predict the last 365 days in the dataset)
train_feature = feature[:-365]
train_label = label[:-365]

test_feature = feature[-365:]
test_label = label[-365:]
#Train and predict using XGBoost
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split

# We will try with 1000 trees and a maximum depth of each tree to be 5
# Early stop if the model hasn't improved in 100 rounds
model = XGBRegressor(
    max_depth=6 # Default - 6
    ,n_estimators=1000
    ,booster='gbtree'
    ,colsample_bytree=1 # Subsample ratio of columns when constructing each tree - default 1
    ,eta=0.3 # Learning Rate - default 0.3
    ,importance_type='gain' # Default is gain
)
model.fit(
    train_feature
    ,train_label
    ,eval_set=[(train_feature, train_label)]
    ,eval_metric='mae'
    ,verbose=True
    ,early_stopping_rounds=100 # Stop after 100 rounds if it doesn't after 100 times
)

xtest['PJME_MW'] = model.predict(test_feature)
XGB_prediction_no_lag = xtest[['Datetime', 'PJME_MW Prediction']].set_index('Datetime')
XGB_prediction_no_lag = XGB_prediction_no_lag.rename(columns={'PJME_MW Prediction': 'PJME_MW'})
# Let's visually see the results
plt.scatter(x=train.index, y=train['PJME_MW'], label='Training Data', color='black')
plt.scatter(x=test.index, y=test['PJME_MW'], label='Test Actuals', color='red')
plt.plot(XGB_prediction_no_lag, label='XGB No Lag', color='orange')

# Plot Labels, Legends etc
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')
plt.title('XGBoost Model (Daily) - Results')

# For clarify, let's limit to only 2015 onwards
plt.xlim(datetime(2015, 1, 1),datetime(2018, 10, 1))

plt.show()
import xgboost as xgb

xgb.plot_importance(model, max_num_features=10, importance_type='gain') # gain is how much each feature contributed to 'improvement' of tree

plt.show()
# First construct the residuals - basically the errors
xgboost_errors = XGB_prediction_no_lag.copy()
xgboost_errors['PJME_MW_ACTUAL'] = test.copy()
xgboost_errors['error'] = xgboost_errors['PJME_MW Prediction'] - xgboost_errors['PJME_MW_ACTUAL']
# Let's visually see the errors via scatterplot
plt.scatter(xgboost_errors.index, xgboost_errors['error'], label='Residual/Errors')

# Plot Labels, Legends etc
plt.title('XGBoost - Residual/Errors Distribution')
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')

plt.show()
# Plot Histogram with Kernel Density Estimation (KDE)
sns.distplot(xgboost_errors['error'], kde=True);

# Plot Labels, Legends etc
plt.title('XGBoost - Residual/Errors Distribution')
from statsmodels.graphics.tsaplots import plot_acf

# Plot the acf function
plot_acf(xgboost_errors['error'],lags=300) #alpha 1 suppresses CI

plt.title('XGBoost - Residual/Errors Autocorrelation')
plt.show()
# So because we need the lag data, we need to preprocess then do the split
all_data = daily_data.copy()

# Create train test dataset using XGBoost preprocessing (365 days top 720 days lag)
feature, label = preprocess_xgb_data(all_data, lag_start=365, lag_end=720)

# We will aim for a 12 month forecast horizon (ie predict the last 365 days in the dataset)
train_feature = feature[:-365]
train_label = label[:-365]

test_feature = feature[-365:]
test_label = label[-365:]

train_feature = train_feature.fillna(0)
test_feature = test_feature.fillna(0)

# train_feature.drop(columns=['date_epoch']) #Don't need timestamp
# test_feature.drop(columns=['date_epoch']) #Don't need timestamp

# Scale dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_feature_scaled = scaler.fit_transform(train_feature)
test_feature_scaled = scaler.transform(test_feature)
# Create Time Series k-fold cross validation
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5) # in this case 5-fold

#Train and predict using LASSO
from sklearn.linear_model import LassoCV

model = LassoCV(
    alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,0.3, 0.6, 1]
    ,max_iter=1000 # 1000 iterations
    ,random_state=42
    ,cv=tscv
    ,verbose=True
)
model.fit(
    train_feature_scaled
    ,train_label
)
LASSO_prediction = xtest.copy()
LASSO_prediction['PJME_MW Prediction'] = model.predict(test_feature_scaled)
LASSO_prediction = LASSO_prediction[['Datetime', 'PJME_MW Prediction']].set_index('Datetime')
LASSO_prediction = LASSO_prediction.rename(columns={'PJME_MW Prediction': 'PJME_MW'})

LASSO_prediction
# Let's visually see the results
plt.scatter(x=train.index, y=train['PJME_MW'], label='Training Data', color='black')
plt.scatter(x=test.index, y=test['PJME_MW'], label='Test Actuals', color='red')
plt.plot(LASSO_Prediction, label='LASSO', color='orange')

# Plot Labels, Legends etc
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')
plt.title('LASSO Model (Daily) - Results')
plt.tight_layout()
plt.grid(True)


# For clarify, let's limit to only 2015 onwards
plt.xlim(datetime(2015, 1, 1),datetime(2018, 10, 1))         

plt.show()
# Plot feature importance by way of coefficients    

# Create DataFrame
coefs = pd.DataFrame(model.coef_, train_feature.columns)
coefs.columns = ["coef"]

# Only grab the Top 10 Coefficients
coefs["abs"] = coefs.coef.apply(np.abs)
coefs = coefs.sort_values(by="abs", ascending=False).head(10)
coefs = coefs.drop(["abs"], axis=1)

# Plot
coefs.coef.plot(kind='bar')

# Plot title and x-axis line
plt.title("Coefficients - Feature Importance")
plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
# First construct the residuals - basically the errors
lasso_errors = LASSO_prediction.copy()
lasso_errors['PJME_MW_ACTUAL'] = test.copy()
lasso_errors['error'] = lasso_errors['PJME_MW'] - lasso_errors['PJME_MW_ACTUAL']
# Plot Histogram with Kernel Density Estimation (KDE)
sns.distplot(lasso_errors['error'], kde=True)

# Plot Labels, Legends etc
plt.title('LASSO - Residual/Errors Distribution')

plt.show()
# Plot the acf function
plot_acf(lasso_errors['error'],lags=300) #alpha 1 suppresses CI

plt.title('LASSO - Residual/Errors Autocorrelation')
plt.show()
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import meanabs

# Equivalent to R's Auto ARIMA to get the optimal parameters
#import pmdarima as pm
#model = pm.auto_arima(htrain, seasonal=True, stationary=True, stepwise=True, trace=True, suppress_warnings=True)

# First we split it up between train and test
htrain = train['PJME_MW'] # SARIMAX takes series, not DF
htest = test['PJME_MW'] # SARIMAX takes series, not DF

# Next define hyperparameters. Default is AR model (1,0,0)(0,0,0,0)
p = 1 # AR order
d = 0 # I degree
q = 1 # MA window
P = 0 # AR seasonal order
D = 1 # I seasonal order
Q = 2 # MA seasonal order
m = 6 # Seasonality period length

model = SARIMAX(
    htrain,
    order=(p, d, q),
    seasonal_order=(P, D, Q, m)
    ,enforce_stationarity=False
    ,enforce_invertibility=False
).fit(
    maxiter=50 # Default is 50
)

results = model.get_prediction(start=htest.index[0], end=htest.index[-1], dynamic=False)
SARIMA_prediction_CI = results.conf_int(alpha=(1-0.8)) # 80% CI
SARIMA_prediction = results.predicted_mean
SARIMA_prediction = SARIMA_prediction.to_frame().rename(columns={0: 'PJME_MW'})

# Evaluate it's performance using Mean Absolute Error (MAE)
print("Finished training and predicting. MAE SARIMA: {:.20f}. AIC: {}. Parameters: p,d,q,P,D,Q,m: ".format(meanabs(htest, SARIMA_prediction['PJME_MW']), model.aic), p,d,q,P,D,Q,m)
# Let's see what the model did
model.plot_diagnostics(figsize=(15, 12))
plt.show()
# Evaluate it's performance using Mean Absolute Error (MAE)
from statsmodels.tools.eval_measures import meanabs

print("MAE SARIMA: {:.20f}".format(meanabs(htest, SARIMA_prediction['PJME_MW'])))
# Let's visually see the results
plt.scatter(x=train.index, y=train['PJME_MW'], label='Training Data', color='black')
plt.scatter(x=test.index, y=test['PJME_MW'], label='Test Actuals', color='red')
plt.plot(SARIMA_prediction['PJME_MW'], label='SARIMA', color='orange')

# Plot Confidence Interval
plt.fill_between(
    SARIMA_prediction.index,
    SARIMA_prediction_CI['lower PJME_MW'],
    SARIMA_prediction_CI['upper PJME_MW'],
    color='skyblue',
    alpha=0.7, # 70% transparency
    label='80% CI'
)

# Plot Labels, Legends etc
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')
plt.title('SARIMA Model (Daily) - Results')

# For clarify, let's limit to only 2017 onwards
plt.xlim(datetime(2017, 1, 1),datetime(2018, 9, 1))

plt.show()
from fbprophet import Prophet

ftrain = train.reset_index().rename(columns={'Datetime':'ds', 'PJME_MW': 'y'}) # Prophet takes ds and y as column names only

model = Prophet(
    n_changepoints=25 # Default is 25
    ,changepoint_prior_scale=0.05 # Default is 0.05
    ,seasonality_mode='additive'
    ,interval_width=0.8 # CI - default is 0.8 or 80%
)
model.fit(ftrain)

# Create the future dataframe with date range that will be used to test accuracy
future_df = test.reset_index()['Datetime'].to_frame().rename(columns={"Datetime":'ds'})

# Predict the future
forecast = model.predict(future_df)
PROPHET_prediction = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']][cutoff:]
PROPHET_prediction = PROPHET_prediction.rename(columns={'yhat': 'PJME_MW'})

print("Finished training and predicting")
model.plot(forecast)

plt.show()
# Let's visually see the results
plt.scatter(x=train.index, y=train['PJME_MW'], label='Training Data', color='black')
plt.scatter(x=test.index, y=test['PJME_MW'], label='Test Actuals', color='red')
plt.plot(PROPHET_prediction['PJME_MW'], label='Prophet', color='orange')

# Plot Confidence Interval
plt.fill_between(
    PROPHET_prediction.index,
    PROPHET_prediction['yhat_lower'],
    PROPHET_prediction['yhat_upper'],
    color='skyblue',
    alpha=0.7, # 70% transparency
    label='80% CI'
)

# Plot Labels, Legends etc
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')
plt.title('Prophet Model (Daily) - Results')

# For clarify, let's limit to only 2017 onwards
plt.xlim(datetime(2017, 1, 1),datetime(2018, 9, 1))

plt.show()
model.plot_components(forecast)

plt.show()
# First construct the residuals - basically the errors
prophet_errors = PROPHET_prediction.copy()
prophet_errors['PJME_MW_ACTUAL'] = test['PJME_MW']
prophet_errors['error'] = prophet_errors['PJME_MW'] - prophet_errors['PJME_MW_ACTUAL']
# Let's visually see the errors via scatterplot
plt.scatter(prophet_errors.index, prophet_errors['error'], label='Residual/Errors')

# Plot Labels, Legends etc
plt.title('Prophet - Residual/Errors Distribution')
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')

plt.show()
# Plot Histogram with Kernel Density Estimation (KDE)
sns.distplot(prophet_errors['error'], kde=True);

# Plot Labels, Legends etc
plt.title('Prophet - Residual/Errors Distribution')
from statsmodels.graphics.tsaplots import plot_acf

# Plot the acf function
plot_acf(xgboost_errors['error'],lags=300) #alpha 1 suppresses CI

plt.title('XGBoost - Residual/Errors Autocorrelation')
plt.show()
# Let's get the top 10 over forecasts (i.e. where the error is the highest negative number)
top_10_errors = prophet_errors.sort_values('error', ascending=True)[['PJME_MW_ACTUAL', 'PJME_MW', 'error']].head(10)

plt.scatter(top_10_errors.index, top_10_errors['PJME_MW'], label='Predicted MW')
plt.scatter(top_10_errors.index, top_10_errors['PJME_MW_ACTUAL'], label='Actual MW')

# Labels, Titles, etc.
plt.title('Prophet - Top 10 Errors')
plt.xlabel("Timestamp")
plt.ylabel("MWh")
plt.legend(loc='best')
top_10_errors.head(10)
print("MAE Baseline: {:.2f}".format(meanabs(test['PJME_MW'], baseline_prediction['PJME_MW'])))
print("MAE HoltWinters: {:.2f}".format(meanabs(test['PJME_MW'], HWES_prediction['PJME_MW'])))
print("MAE XGBoost: {:.2f}".format(mean_absolute_error(test_label, XGB_prediction_no_lag['PJME_MW'])))
print("MAE LASSO: {:.2f}".format(mean_absolute_error(test_label, LASSO_prediction['PJME_MW'])))
print("MAE SARIMA: {:.2f}".format(meanabs(test['PJME_MW'], SARIMA_prediction['PJME_MW'])))
print("MAE Prophet: {:.2f}".format(meanabs(test['PJME_MW'], PROPHET_prediction['PJME_MW'])))
def MAPE(y_true, y_pred): 
    '''Function to calculate MAPE'''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print("MAE Baseline: {:.2f}%".format(MAPE(test['PJME_MW'], baseline_prediction['PJME_MW'])))
print("MAE HoltWinters: {:.2f}%".format(MAPE(test['PJME_MW'], HWES_prediction['PJME_MW'])))
print("MAE XGBoost: {:.2f}%".format(MAPE(test['PJME_MW'], XGB_prediction_no_lag['PJME_MW'])))
print("MAE LASSO: {:.2f}%".format(MAPE(test['PJME_MW'], LASSO_prediction['PJME_MW'])))
print("MAE SARIMA: {:.2f}%".format(MAPE(test['PJME_MW'], SARIMA_prediction['PJME_MW'])))
print("MAE Prophet: {:.2f}%".format(MAPE(test['PJME_MW'], PROPHET_prediction['PJME_MW'])))
def plot_model_result(ax, prediction, model_name, color):
    '''
    Plot model results.
    
    prediction : DataFrame
    model_name : str
    
    return ax
    '''
    # Training and Test Actuals
    ax.scatter(x=train.index, y=train['PJME_MW'], label='Training Data', color='black')
    ax.scatter(x=test.index, y=test['PJME_MW'], label='Test Actuals', color='red')

    # Model Results
    ax.plot(prediction['PJME_MW'], label=model_name, color=color, alpha=0.7)
    
    # For clarify, let's limit to only August 2016 onwards
    ax.set_xlim(datetime(2016, 8, 1),datetime(2018, 10, 1))

    # Set Y Axis
    ax.set_ylim(500000, 1100000)
    
    # Set Axis labels
    ax.set_ylabel("MWh")
    ax.legend(loc='best')
    ax.set_title(
        "{}: MAPE: {:.2f}% | MAE: {:.2f}".format(
            model_name,MAPE(test["PJME_MW"], prediction["PJME_MW"]),
            mean_absolute_error(test["PJME_MW"], prediction["PJME_MW"])
            )
        , fontsize=40
    )

    return ax
fig, ax = plt.subplots(6,1, figsize=(30,40))

# Plot Labels, Legends etc
plt.xlabel("Timestamp")

ax[0] = plot_model_result(ax[0], baseline_prediction, model_name='Baseline (Naive)', color='midnightblue')
ax[1] = plot_model_result(ax[1], XGB_prediction_no_lag, model_name='XGBoost', color='deepskyblue')
ax[2] = plot_model_result(ax[2], PROPHET_prediction, model_name='Prophet', color='steelblue')
ax[3] = plot_model_result(ax[3], LASSO_prediction, model_name='LASSO', color='royalblue')
ax[4] = plot_model_result(ax[4], HWES_prediction, model_name='Holtwinters', color='lightsteelblue')
ax[5] = plot_model_result(ax[5], SARIMA_prediction, model_name='SARIMA', color='dodgerblue')

plt.tight_layout(pad=3.0)
plt.show()