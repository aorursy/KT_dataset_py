import pandas as pd
import numpy as np
import math
import pickle
from scipy.stats import kruskal, pearsonr, randint, uniform, chi2_contingency, boxcox
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer, StandardScaler, power_transform
from sklearn.linear_model import SGDClassifier
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, cross_validate, TimeSeriesSplit, RandomizedSearchCV, GridSearchCV, cross_val_predict
from datetime import datetime
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss, acf, pacf
from collections import defaultdict, OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import AR
from sklearn.linear_model import LinearRegression
import xgboost as xgb

import seaborn as sb
import matplotlib.pyplot as plt 

%matplotlib inline

# diplaying all columns without truncation in dataframes
pd.set_option('display.max_columns', 500)

# read in bike sharing dataset
bike_df = pd.read_csv('/kaggle/input/bike-sharing-washington-dc/bike_sharing_dataset.csv')
bike_df.head()

# print descriptive statistics
bike_df.describe()

# check the datatypes of each variable
bike_df.dtypes

# Check for missing values
bike_df.isnull().sum()

# fill NAs with 0 where applicable
wt_feats = [x for x in bike_df.columns if 'wt' in x]
bike_df['holiday'] = bike_df['holiday'].fillna(0)
bike_df[wt_feats] = bike_df[wt_feats].fillna(0)

# check casual, registered and total_cust missing rows
missing_target = bike_df[bike_df['total_cust'].isna()]
missing_target

# filling the missing values in the customer variables with forward fill method
bike_df[['total_cust', 'casual', 'registered']] = bike_df[['total_cust', 'casual', 'registered']].fillna(
                                                            method='ffill')
bike_df.isnull().sum()

# check what the correlation between the different temperature features and total_cust is
# maybe I could just use one of the other temperature features instead of temp_avg

# correlation between temp_avg and total_cust
# I'm excluding the first 820 rows because they contain missing temp_avg values
print('temp_avg:', pearsonr(bike_df['temp_avg'][821:], bike_df['total_cust'][821:]))

# correlation between temp_min and total_cust
print('temp_min:', pearsonr(bike_df['temp_min'], bike_df['total_cust']))
print('temp_min, without first 820 rows:', pearsonr(bike_df['temp_min'][821:], bike_df['total_cust'][821:]))

# correlation between temp_max and total_cust
print('temp_max:', pearsonr(bike_df['temp_max'], bike_df['total_cust']))
print('temp_max, without first 820 rows:', pearsonr(bike_df['temp_max'][821:], bike_df['total_cust'][821:]))

# correlation between temp_observ and total_cust
print('temp_observ:', pearsonr(bike_df['temp_observ'], bike_df['total_cust']))
print('temp_observ, without first 820 rows:', pearsonr(bike_df['temp_observ'][821:], bike_df['total_cust'][821:]))

# calculating the Granger causality between the temperature feats and the target
print('Average temperature and target:')
grangercausalitytests(bike_df[['total_cust', 'temp_avg']][821:], maxlag=1);
print('\nMaximum temperature and target:')
grangercausalitytests(bike_df[['total_cust', 'temp_max']], maxlag=1);
print('\nMinimum temperature and target:')
grangercausalitytests(bike_df[['total_cust', 'temp_min']], maxlag=1);
print('\nObserved temperature and target:')
grangercausalitytests(bike_df[['total_cust', 'temp_observ']], maxlag=1);

# function to create seasons for dataframe
def seasons(df):
    '''
    Function to create new features for seasons based on months
    Args: df = dataframe
    Returns: df = dataframe
    '''
    
    # create a season features
    df['season_spring'] = df['date'].apply(lambda x: 1 if '01' in x[5:7] else 1 if '02' in x[5:7] else 1 
                                                     if '03' in x[5:7] else 0)
    df['season_summer'] = df['date'].apply(lambda x: 1 if '04' in x[5:7] else 1 if '05' in x[5:7] else 1 
                                                     if '06' in x[5:7] else 0)
    df['season_fall'] = df['date'].apply(lambda x: 1 if '07' in x[5:7] else 1 if '08' in x[5:7] else 1 
                                                     if '09' in x[5:7] else 0)
    
    return df

### create new features for seasons
bike_df = seasons(bike_df)

### create new feature weekday
bike_df['date_datetime'] = bike_df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

bike_df['weekday'] = bike_df['date_datetime'].apply(lambda x: x.weekday())

### one hot encode the feature weekday
weekday_dummies = pd.get_dummies(bike_df['weekday'], prefix='weekday', drop_first=True)
bike_df = bike_df.join(weekday_dummies, how='left')
bike_df.head()

### create new feature working_day
bike_df['working_day'] = bike_df['weekday'].apply(lambda x: 0 if x > 5 or x == 0 else 1)
bike_df['working_day'] = bike_df[['holiday', 'working_day']].apply(
    lambda x: 0 if x['holiday'] == 1 else x['working_day'], axis=1)

# Dropping date, registered and casual features because this is in string format 
bike_df.drop(columns=['date', 'temp_avg', 'registered', 'casual'], inplace=True)

# variable to be used below to iterate through the columns and plot them
season_names = ['season_spring', 'season_summer', 'season_fall']

# plot boxplots for season versus number of users
plt.figure(figsize = [15, 5])

# boxplot for feature workingday
plt.subplot(1, 3, 1)
sb.boxplot(data = bike_df, x = 'season_spring', y = 'total_cust')
plt.xlabel('Spring')
plt.ylabel('Number of customers')

# boxplot for feature weekday
plt.subplot(1, 3, 2)
sb.boxplot(data = bike_df, x = 'season_summer', y = 'total_cust')
plt.xlabel('Summer')
plt.ylabel('Number of customers')

# boxplot for feature holiday
plt.subplot(1, 3, 3)
sb.boxplot(data = bike_df, x = 'season_fall', y = 'total_cust')
plt.xlabel('Fall')
plt.ylabel('Number of customers');

# Correlation between season features and the maximum temperature
# using the Kruskal Wallis H test for correlations between a continuous and categorical variable
kruskal(bike_df['temp_max'], bike_df['season_summer'])

# plotting the customer statistics in form of a boxplot for the holiday feature
sb.boxplot(data = bike_df, x = 'holiday', y = 'total_cust');

# Correlation between holiday feature and the number of customers per day
# using the Kruskal Wallis H test for correlations between a continuous and categorical variable
kruskal(bike_df['holiday'], bike_df['total_cust'])

# plotting the customer statistics in form of a boxplot for the weekday feature
plt.figure(figsize = [15, 5])
sb.boxplot(data = bike_df, x = 'weekday', y = 'total_cust')
plt.xlabel('Weekday')
plt.ylabel('Number of customers');

# Correlation between weekday feature and the number of customers per day
# using Pearson's correlation coefficient because I'm assuming that weekday can be 
# considered a continuous variable
pearsonr(bike_df['weekday'], bike_df['total_cust'])

# plotting workday feature in boxplot against the count of customers
sb.boxplot(data = bike_df, x = 'working_day', y = 'total_cust')
plt.xlabel('Working day or not')
plt.ylabel('Number of customers');

# Correlation between working_day feature and the number of customers per day
# using the Kruskal Wallis H test for correlations between a continuous and categorical variable
kruskal(bike_df['working_day'], bike_df['total_cust'])

# create a new dataframe that encodes the weekday feature with 0 for monday through friday
# and 1 for saturday and sunday
weekend_distinct_df = bike_df.copy()
weekend_distinct_df['weekday'] = weekend_distinct_df['weekday'].apply(lambda x: 1 if (x == 6 or x == 0) else 0)

# plot boxplots for comparison between the weekday and workingday feature
plt.figure(figsize = [15, 5])

# boxplot for feature workingday
plt.subplot(1, 3, 1)
sb.boxplot(data = bike_df, x = 'working_day', y = 'total_cust');

# boxplot for feature weekday
plt.subplot(1, 3, 2)
sb.boxplot(data = weekend_distinct_df, x = 'weekday', y = 'total_cust');

# boxplot for feature holiday
plt.subplot(1, 3, 3)
sb.boxplot(data = bike_df, x = 'holiday', y = 'total_cust');

# plot the means of each instance of workingday
bike_df.groupby('working_day')['total_cust'].mean()

# plot the means of each instance of weekday
weekend_distinct_df.groupby('weekday')['total_cust'].mean()

# plot the means of each instance of holiday
bike_df.groupby('holiday')['total_cust'].mean()

# plotting the revenue of the most common production companies vs. the rest
fig, ax = plt.subplots(6, 3, figsize = [16, 25])

# create list with all feature names 
wt_feat_list = [x for x in bike_df.columns if 'wt_' in x]

# company counter
counter = 0

for j in range(len(ax)):
    for i in range(len(ax[j])):
        if j == 5 and i == 3:
            break
        else:
            ax[j][i] = sb.boxplot(data = bike_df, x = wt_feat_list[counter], y = 'total_cust', ax=ax[j][i])
            ax[j][i].set_ylabel('Number of customers')
            ax[j][i].set_xlabel(wt_feat_list[counter])
            counter += 1
            
# fog, heavy fog, hail, haze, high wind
bike_df['foggy'] = bike_df['wt_fog'] + bike_df['wt_heavy_fog'] + bike_df['wt_hail'] + bike_df['wt_haze'] + bike_df['wt_high_wind']
bike_df['foggy'] = bike_df['foggy'].apply(lambda x: 0 if x == 0 else 1)

# thunder
bike_df['thunder'] = bike_df['wt_thunder']

# ice_fog, unknown, freeze_drizzle, freeze_rain, drift_snow
bike_df['ice'] = bike_df['wt_ice_fog'] + bike_df['wt_unknown'] + bike_df['wt_freeze_drizzle'] + bike_df['wt_freeze_rain'] + bike_df['wt_drift_snow']
bike_df['ice'] = bike_df['ice'].apply(lambda x: 0 if x == 0 else 1)

# sleet, glaze, snow
bike_df['sleet'] = bike_df['wt_sleet'] + bike_df['wt_glaze'] + bike_df['wt_snow']
bike_df['sleet'] = bike_df['sleet'].apply(lambda x: 0 if x == 0 else 1)

# mist, drizzle, rain, ground fog
bike_df['rain'] = bike_df['wt_mist'] + bike_df['wt_drizzle'] + bike_df['wt_rain'] + bike_df['wt_ground_fog']
bike_df['rain'] = bike_df['rain'].apply(lambda x: 0 if x == 0 else 1)


# drop the old wt features
bike_df.drop(columns=wt_feats, inplace=True)

# plot boxplots for comparison between the new weather type features and target
plt.figure(figsize = [12, 10])

# boxplot for feature foggy
plt.subplot(2, 2, 1)
sb.boxplot(data = bike_df, x = 'foggy', y = 'total_cust')
plt.xlabel('Weather type foggy')
plt.ylabel('Number of customers')

# boxplot for feature thunder
plt.subplot(2, 2, 2)
sb.boxplot(data = bike_df, x = 'thunder', y = 'total_cust')
plt.xlabel('Weather type thunder')
plt.ylabel('Number of customers')

# boxplot for feature ice
plt.subplot(2, 2, 3)
sb.boxplot(data = bike_df, x = 'ice', y = 'total_cust')
plt.xlabel('Weather type ice')
plt.ylabel('Number of customers')

# boxplot for feature sleet
plt.subplot(2, 2, 4)
sb.boxplot(data = bike_df, x = 'sleet', y = 'total_cust')
plt.xlabel('Weather type sleet')
plt.ylabel('Number of customers');


# plotting the acf of wt_ features
plt.figure(figsize=[15,6])

plot_acf(bike_df['foggy'], title='ACF: All samples in metro DC area',)
plt.show()

plot_acf(bike_df['thunder'], title='ACF: All samples in metro DC area',)
plt.show()

plot_acf(bike_df['ice'], title='ACF: All samples in metro DC area',)
plt.show()

plot_acf(bike_df['sleet'], title='ACF: All samples in metro DC area',)
plt.show()

plot_acf(bike_df['rain'], title='ACF: All samples in metro DC area',)
plt.show()

# plotting all newly created features
plt.figure(figsize=[20,9])

plt.subplot(3,2,1)
plt.plot(bike_df['thunder'])

plt.subplot(3, 2, 2)
plt.plot(bike_df['foggy'])

plt.subplot(3, 2, 3)
plt.plot(bike_df['ice'])

plt.subplot(3, 2, 4)
plt.plot(bike_df['sleet'])

plt.subplot(3, 2, 5)
plt.plot(bike_df['rain'])

bike_df.drop(columns=['rain'], inplace=True)
# get list for all rolling sums between the weather features and the target

def best_window_sum(x, y, max_window):
    corr_temp_cust = []
    for i in range(1, max_window):
        roll_val = list(x.rolling(i).sum()[i-1:-1])
        total_cust_ti = list(y[i:])
        corr, p_val = pearsonr(total_cust_ti, roll_val)
        corr_temp_cust.append(corr)
    # get the optimal window size for rolling mean
    max_val = np.argmax(corr_temp_cust)
    min_val = np.argmin(corr_temp_cust)
    opt_corr_min = corr_temp_cust[min_val]
    opt_corr_max = corr_temp_cust[max_val]
    
    results = {max_val+1: opt_corr_max, min_val+1: opt_corr_min}
    
    return results
    
# get the optimal window for rolling sum for foggy
print(best_window_sum(bike_df['foggy'], bike_df['total_cust'], 30))

# get the correlation for window size determined by foggy
foggy_mean = bike_df['foggy'].rolling(8).sum()[7:-1]
pearsonr(foggy_mean, bike_df['total_cust'][8:])

# get the optimal window for rolling sum for thunder
print(best_window_sum(bike_df['thunder'], bike_df['total_cust'], 30))

# get the correlation for window size determined by thunder
thunder_mean = bike_df['thunder'].rolling(8).sum()[7:-1]
pearsonr(thunder_mean, bike_df['total_cust'][8:])

# get the optimal window for rolling std for ice
print(best_window_sum(bike_df['ice'], bike_df['total_cust'], 30))

# get the correlation for window size determined by ice
ice_mean = bike_df['ice'].rolling(8).sum()[7:-1]
pearsonr(ice_mean, bike_df['total_cust'][8:])

# get the optimal window for rolling std for sleet
print(best_window_sum(bike_df['sleet'], bike_df['total_cust'], 30))

# get the correlation for window size determined by sleet
sleet_mean = bike_df['sleet'].rolling(8).sum()[7:-1]
pearsonr(sleet_mean, bike_df['total_cust'][8:])

# drop any non-categorical variables
bike_df_corr_cat = bike_df.drop(columns=['date_datetime', 'weekday', 'temp_min', 'temp_max',
                                         'temp_observ', 'precip', 'wind', 'total_cust'], axis=1)


# this code snippet was taken from https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# create correlation matrix with cramer's V coefficients
corr_matrix = pd.DataFrame(data = None, index=np.arange(len(bike_df_corr_cat.columns)), 
                            columns=bike_df_corr_cat.columns)

for col in bike_df_corr_cat.columns:
    count = 0
    for val in bike_df_corr_cat.columns:
        corr_cat = cramers_v(bike_df_corr_cat[col], bike_df_corr_cat[val])
        corr_matrix[col][count] = corr_cat
        count += 1
    corr_matrix = corr_matrix.astype('float')


# add an index to the dataframe
corr_matrix['columns'] = bike_df_corr_cat.columns
corr_matrix.set_index('columns', inplace=True)

# plot a heatmap for correlations between categorical variables
plt.figure(figsize=[15,10])
sb.heatmap(corr_matrix, annot=True,
          vmin=-1, vmax=1, center=0,
          fmt='.2g', cmap='YlGnBu')
plt.title('Heatmap of correlations between categorical variables')
plt.xlabel('Columns')
plt.ylabel('Columns');

# plot all distributions and scatterplot between each continuous variable pair
sb.pairplot(bike_df, vars=['temp_min', 'temp_max', 'temp_observ', 'wind', 'precip', 'total_cust']);

# create a correlation matrix
bike_df_corr = bike_df[['temp_min', 'temp_max', 'temp_observ', 'wind', 'precip', 'total_cust']].corr()

# create a heatmap to visualize the results
plt.figure(figsize=[10,6])
sb.heatmap(bike_df_corr, annot=True,
          vmin=-1, vmax=1, center=0,
          cmap='YlGnBu')
plt.title('Heatmap of correlations between continuous variables')
plt.xlabel('Columns')
plt.ylabel('Columns');

# get list for all correlations between a feature and total_cust with different rolling means

def best_window(x, y, max_window):
    corr_temp_cust = []
    for i in range(1, max_window):
        roll_val = list(x.rolling(i).mean()[i-1:-1])
        total_cust_ti = list(y[i:])
        corr, p_val = pearsonr(total_cust_ti, roll_val)
        corr_temp_cust.append(corr)
    # get the optimal window size for rolling mean between a feature and total_cust
    max_val = np.argmax(corr_temp_cust)
    min_val = np.argmin(corr_temp_cust)
    opt_corr_min = corr_temp_cust[min_val]
    opt_corr_max = corr_temp_cust[max_val]
    
    results = {max_val+1: opt_corr_max, min_val+1: opt_corr_min}
    
    return results
    
# get list for all correlations between a feature and total_cust with different rolling standard deviations

def best_window_std(x, y, max_window):
    corr_temp_cust = []
    for i in range(2, max_window):
        roll_val = list(x.rolling(i).std()[i-1:-1])
        total_cust_ti = list(y[i:])
        corr, p_val = pearsonr(total_cust_ti, roll_val)
        corr_temp_cust.append(corr)
    # get the optimal window size for rolling std between a feature and total_cust
    max_val = np.argmax(corr_temp_cust)
    min_val = np.argmin(corr_temp_cust)
    opt_corr_min = corr_temp_cust[min_val]
    opt_corr_max = corr_temp_cust[max_val]
    
    results = {max_val+1: opt_corr_max, min_val+1: opt_corr_min}
    
    return results

# plot the overall total_cust values for entire timeseries
plt.figure(figsize=[15,6])
ax = sb.lineplot(x='date_datetime', y='total_cust', data=bike_df)

# plot only last two years of timeseries
plt.figure(figsize=[15,6])
ax = sb.lineplot(x='date_datetime', y='total_cust', data=bike_df[-718:])

# plotting the partial autocorrelation for target
plot_pacf(bike_df['total_cust'], title='Partical autocorrelation of total_cust',)
plt.xlabel('Number of days in the past')
plt.ylabel('Partial autocorrelation coefficient')
plt.show()

# plotting the autocorrelation for target
plot_acf(bike_df['total_cust'], title='Autocorrelation of total_cust',)
plt.xlabel('Number of days in the past')
plt.ylabel('Autocorrelation coefficient')
plt.show()

# get the optimal window for rolling std for total_cust
print(best_window_std(bike_df['total_cust'], bike_df['total_cust'], 30))

# get the correlation for window size determined by total_cust
cust_mean = bike_df['total_cust'].rolling(8).std()[7:-1]
pearsonr(cust_mean, bike_df['total_cust'][8:])
# get the optimal number for rolling mean for total_cust
print(best_window(bike_df['total_cust'], bike_df['total_cust'], 30))

# get the correlation for window size determined by total_cust
cust_mean = bike_df['total_cust'].rolling(8).mean()[7:-1]
pearsonr(cust_mean, bike_df['total_cust'][8:])

# add the value from t-1
bike_df['total_cust_t-1'] = bike_df['total_cust'].shift()

# create series that group the mean temperature per season
temp_spring = bike_df.groupby('season_spring')['temp_max'].mean().rename({1: 'Spring'})
temp_summer = bike_df.groupby('season_summer')['temp_max'].mean().rename({1: 'Summer'})
temp_fall = bike_df.groupby('season_fall')['temp_max'].mean().rename({1: 'Fall'})

# add them to one series and drop the rows with index 0
temp_seasons = temp_spring.append(temp_summer).append(temp_fall)
temp_seasons.drop(labels=[0], inplace=True)

# plot average temp_max per season
plt.figure(figsize=[10,6])
sb.barplot(x=temp_seasons.index, y=temp_seasons.values);

# create series that groups average users per season
cust_spring = bike_df.groupby('season_spring')['total_cust'].mean().rename({1: 'Spring'})
cust_summer = bike_df.groupby('season_summer')['total_cust'].mean().rename({1: 'Summer'})
cust_fall = bike_df.groupby('season_fall')['total_cust'].mean().rename({1: 'Fall'})

# add them to one series and drop the rows with index 0
cust_seasons = cust_spring.append(cust_summer).append(cust_fall)
cust_seasons.drop(labels=[0], inplace=True)

# assign x and y1 and y2
x = list(temp_seasons.index)
y1 = cust_seasons.values
y2 = temp_seasons.values

# below code adapted from https://matplotlib.org/gallery/api/two_scales.html
# creat plot containing both average count of customers
# and average temp per month
plt.figure(figsize=[15,7])
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color1 = 'tab:red'
ax1.set_xlabel('Seasons')
ax1.set_ylabel('Average number of customers', color=color1)
ax1.bar(x, y1, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

color2 = 'tab:blue'
ax2.set_ylabel('Average maximum temperature', color=color2)
ax2.plot(x, y2, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.show()

# plotting temp feature against the target label total_cust
plt.figure(figsize=[7,5])

sb.scatterplot(data = bike_df, x = 'temp_max', y = 'total_cust', color='green')
plt.xlabel('Maximum temperature')
plt.ylabel('Number of customers')
plt.title('Maximum temperature vs. number of customers');


# plot the partial autocorrelation of temp_max
plot_pacf(bike_df['temp_max'], title='PACF: All samples in metro DC area',)
plt.show()

# autocorrelation of temp_max
plot_acf(bike_df['temp_max'], title='ACF: All samples in metro DC area',)
plt.show()

# get the optimal window for rolling std for temperature
print(best_window_std(bike_df['temp_max'], bike_df['total_cust'], 30))

# get the correlation for window size determined by temp_max
temp_mean = bike_df['temp_max'].rolling(8).std()[7:-1]
pearsonr(temp_mean, bike_df['total_cust'][8:])
# get the optimal window for rolling mean for temperature
best_window(bike_df['temp_max'], bike_df['total_cust'], 30)

# plot the overall temp_max values for entire timeseries
plt.figure(figsize=[15,6])
ax = sb.lineplot(x='date_datetime', y='temp_max', data=bike_df)
plt.ylabel('Maximum temperature [°C]')
plt.xlabel('Time');

# create plot of rolling means
plt.figure(figsize=[15,6])

plt.plot(bike_df['temp_max'].rolling(16).mean());

# create plot of rolling stds
plt.figure(figsize=[15,6])

plt.plot(bike_df['temp_max'].rolling(16).std());

# plotting temp_min feature against the target label
plt.figure(figsize=[7,5])

sb.scatterplot(data = bike_df, x = 'temp_min', y = 'total_cust', color='red')
plt.xlabel('Minimum temperature')
plt.ylabel('Number of customers')
plt.title('Minimum temperature vs. number of customers');


# engineer new categorical temp features
bike_df_cat = bike_df.copy()
bike_df_cat['very_cold'] = bike_df_cat['temp_max'].apply(lambda x: 1 if x < 0 else 0)
bike_df_cat['cold'] = bike_df_cat['temp_max'].apply(lambda x: 1 if x < 10 and x >= 0 else 0)
bike_df_cat['cool'] = bike_df_cat['temp_max'].apply(lambda x: 1 if x < 20 and x >= 10 else 0)
bike_df_cat['warm'] = bike_df_cat['temp_max'].apply(lambda x: 1 if x < 30 and x >= 20 else 0)
bike_df_cat['hot'] = bike_df_cat['temp_max'].apply(lambda x: 1 if x >= 30 else 0)

bike_df_cat[['very_cold', 'cold', 'cool', 'warm', 'hot']] = bike_df_cat[['very_cold', 'cold', 'cool', 'warm', 'hot']].shift()


bike_df_cat = bike_df_cat.iloc[1:,:]
pearsonr(bike_df_cat['total_cust'], bike_df_cat['cold'])
# plot categorized temperature
plt.figure(figsize=[15,10])
plt.subplot(2,3,1)
sb.boxplot(bike_df_cat['very_cold'], bike_df_cat['total_cust'])
plt.xlabel('Very cold')
plt.ylabel('Number of customers')

plt.subplot(2,3,2)
sb.boxplot(bike_df_cat['cold'], bike_df_cat['total_cust'])
plt.xlabel('Cold')
plt.ylabel('Number of customers')

plt.subplot(2,3,3)
sb.boxplot(bike_df_cat['cool'], bike_df_cat['total_cust'])
plt.xlabel('Cool')
plt.ylabel('Number of customers')

plt.subplot(2,3,4)
sb.boxplot(bike_df_cat['warm'], bike_df_cat['total_cust'])
plt.xlabel('Warm')
plt.ylabel('Number of customers')

plt.subplot(2,3,5)
sb.boxplot(bike_df_cat['hot'], bike_df_cat['total_cust'])
plt.xlabel('Hot')
plt.ylabel('Number of customers')

# plotting the distribution of precip
plt.figure(figsize=[15,6])

plt.subplot(1,2,1)
plt.hist(bike_df['precip'], bins=15)
plt.title('Distribution of precip')

plt.subplot(1,2,2)
sb.scatterplot(data = bike_df, x = 'precip', y = 'total_cust')
plt.xlabel('Precipitation')
plt.ylabel('Number of customers')
plt.title('Precipitation vs. number of customers');

# plotting the distribution of precip
plt.figure(figsize=[10,6])

x = np.log10(bike_df['precip'] + 1)
plt.hist(x)
plt.title('Distribution of precip');

# plot the overall precip values for entire timeseries
plt.figure(figsize=[15,6])
ax = sb.lineplot(x='date_datetime', y='precip', data=bike_df)

# get the optimal number for rolling mean window
print(best_window(bike_df['precip'], bike_df['total_cust'], 30))

# get the correlation for window size determined by precip
precip_mean = bike_df['precip'].rolling(8).mean()[7:-1]
pearsonr(precip_mean, bike_df['total_cust'][8:])

# get the optimal window for rolling std for precip
print(best_window_std(bike_df['precip'], bike_df['total_cust'], 30))

# get the correlation for window size determined by precip
precip_mean = bike_df['precip'].rolling(8).std()[7:-1]
pearsonr(precip_mean, bike_df['total_cust'][8:])

# plot the partial autocorrelation of precip
plot_pacf(bike_df['precip'], title='PACF: All samples in metro DC area',)
plt.show()

# create plot of rolling means
plt.figure(figsize=[15,6])

plt.plot(bike_df['precip'].rolling(16).mean());

# create plot of rolling stds
plt.figure(figsize=[15,6])

plt.plot(bike_df['precip'].rolling(16).std());

# plotting the distribution of wind
plt.figure(figsize=[15,6])

plt.subplot(1,2,1)
plt.hist(bike_df['wind'], bins=15)
plt.title('Distribution of wind')

plt.subplot(1,2,2)
sb.scatterplot(data = bike_df, x = 'wind', y = 'total_cust')
plt.xlabel('Wind')
plt.ylabel('Number of customers')
plt.title('Wind vs. number of customers');


# plot the overall wind values for entire timeseries
plt.figure(figsize=[15,6])
ax = sb.lineplot(x='date_datetime', y='wind', data=bike_df)

# get the optimal number for rolling mean window
print(best_window(bike_df['wind'], bike_df['total_cust'], 30))

# get the correlation for window size determined by wind
wind_mean = bike_df['wind'].rolling(8).mean()[7:-1]
pearsonr(wind_mean, bike_df['total_cust'][8:])[0]

# get the optimal window for rolling std for temperature
print(best_window_std(bike_df['wind'], bike_df['total_cust'], 30))

# get the correlation for window size determined by wind
precip_mean = bike_df['wind'].rolling(8).std()[7:-1]
pearsonr(precip_mean, bike_df['total_cust'][8:])

# plot the partial autocorrelation of wind
plot_pacf(bike_df['wind'], title='PACF: All samples in metro DC area',)
plt.show()

# create plot of rolling means
plt.figure(figsize=[15,6])

plt.plot(bike_df['wind'].rolling(16).mean());

# create plot of rolling means
plt.figure(figsize=[15,6])

plt.plot(bike_df['wind'].rolling(16).std());

# code based on implementation on https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
def adf_test(df, col_names):
    '''
    Function to perform Augmented Dickey-Fuller test on selected timeseries
    Args: df = dataframe with timeseries to be tested
          col_names = list of names of the timeseries to be tested
    Returns: None
    '''
    for name in col_names:
        print ('Results of Augmented Dickey-Fuller Test for {}'.format(name))
        result_test = adfuller(df[name], autolag='AIC')
        result_output = pd.Series(result_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key, val in result_test[4].items():
            result_output['Critical Value (%s)'%key] = val
        print (result_output)

# create the features that need to be tested
# total_cust_t-1 was already added to the dataframe

testing_feat = ['wind', 'precip', 'total_cust', 'temp_min', 'temp_max', 'foggy', 'ice', 'thunder', 'sleet']

testing_df = pd.DataFrame()

for col in testing_feat:
    col_mean = bike_df[col].rolling(16).mean()[15:-1]
    col_std = bike_df[col].rolling(16).std()[15:-1]
    testing_df[col+'_mean16'] = col_mean.values
    testing_df[col+'_std16'] = col_std.values

# adf test for total_cust_t-1
temp_cust_1 = bike_df['total_cust_t-1'].fillna(0)
bike_df_temp = pd.DataFrame(temp_cust_1, columns=['total_cust_t-1'])
adf_test(bike_df_temp, ['total_cust_t-1'])

# adf test for total_cust
adf_test(bike_df, ['total_cust'])

# adf test for all engineered features
adf_test(testing_df, testing_df.columns)

# code based on implementation on https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
def kpss_test(df, col_names):
    '''
    Function to perform KPSS test on selected timeseries
    Args: df = dataframe with timeseries to be tested
          col_names = list of names of the timeseries to be tested
    Returns: None
    '''
    for name in col_names:
        print ('Results of KPSS Test for {}'.format(name))
        result_test = kpss(df[name], regression='c', lags='legacy')
        result_output = pd.Series(result_test[0:3], index=['Test Statistic','p-value','Lags Used'])
        for key, val in result_test[3].items():
            result_output['Critical Value (%s)'%key] = val
        print (result_output)

# kpss test for total_cust
kpss_test(testing_df, testing_df.columns)

# kpss test for total_cust_t-1
kpss_test(bike_df_temp, ['total_cust_t-1'])

# kpss test for total_cust
kpss_test(bike_df, ['total_cust'])

# Removing positive upward trend from total_cust_mean16, total_cust_std16 and total_cust_t-1
testing_df['total_cust_std16_log'] = [np.log1p(x+1) for x in testing_df['total_cust_std16']]
testing_df['total_cust_mean16_log'] = [np.log1p(x+1) for x in testing_df['total_cust_mean16']]
bike_df_temp['total_cust_t-1_log'] = [np.log1p(x+1) for x in bike_df_temp['total_cust_t-1']]

# applying differencing to remove trend from total_cust
bike_df_check = bike_df[['total_cust']]
bike_df_check['total_cust_diff'] = bike_df_check['total_cust'] - bike_df_check['total_cust'].shift()
bike_df_check = bike_df_check.iloc[1:,]
# kpss test for total_cust
kpss_test(testing_df, ['total_cust_std16_log', 'total_cust_mean16_log'])
kpss_test(bike_df_temp, ['total_cust_t-1_log'])

# adf test for total_cust
adf_test(testing_df, ['total_cust_std16_log', 'total_cust_mean16_log'])
adf_test(bike_df_temp, ['total_cust_t-1_log'])
adf_test(bike_df_check, ['total_cust_diff'])

# should keep this before the cleaning function has been created
bike_df.drop(columns=['temp_observ', 'weekday'], axis=1, inplace=True)

# drop the timestamp variable
bike_df.drop(columns=['date_datetime'], inplace=True)

# specify the window for rolling values
window = 8

# engineer new categorical temp features
bike_df_cat = bike_df[['temp_max']].copy()
bike_df_cat['very_cold'] = bike_df_cat['temp_max'].apply(lambda x: 1 if x < 0 else 0)
bike_df_cat['cold'] = bike_df_cat['temp_max'].apply(lambda x: 1 if x < 10 and x >= 0 else 0)
bike_df_cat['cool'] = bike_df_cat['temp_max'].apply(lambda x: 1 if x < 20 and x >= 10 else 0)
bike_df_cat['warm'] = bike_df_cat['temp_max'].apply(lambda x: 1 if x < 30 and x >= 20 else 0)
bike_df_cat['hot'] = bike_df_cat['temp_max'].apply(lambda x: 1 if x >= 30 else 0)

bike_df_cat[['very_cold', 'cold', 'cool', 'warm', 'hot']] = bike_df_cat[['very_cold', 'cold', 'cool', 'warm', 'hot']].shift()

bike_df_cat.drop(columns=['temp_max'], inplace=True)

# creating rolling values
new_feat = ['wind', 'precip', 'total_cust', 'temp_max', 'temp_min', 'foggy', 'ice', 'thunder', 'sleet']

temp_df = pd.DataFrame()

for col in new_feat:
    col_mean = bike_df[col].rolling(window).mean()[(window-1):-1]
    col_std = bike_df[col].rolling(window).std()[(window-1):-1]
    temp_df[col+'_mean'+str(window)] = col_mean.values
    temp_df[col+'_std'+str(window)] = col_std.values

# remember to remove the first row from 16 rows from total_cust (target label)
new_bike_df = bike_df.iloc[window:,:]
bike_df_cat = bike_df_cat.iloc[window:,:]
new_bike_df.reset_index(drop=True, inplace=True)
bike_df_cat.reset_index(drop=True, inplace=True)
new_bike_df.head()

# drop all columns which we cannot use because of lookahead bias
new_bike_df.drop(columns=['temp_max', 'wind', 'temp_min', 'precip', 
                          'thunder', 'foggy', 'ice', 'sleet'], inplace=True)

# merging both dataframes with features
final_bike_df = new_bike_df.join(temp_df, how='left')
final_bike_df = final_bike_df.join(bike_df_cat, how='left')
final_bike_df.head()

# assigning X and y
y = final_bike_df['total_cust']
X = final_bike_df.drop(columns=['total_cust'])

# creating a class that I can use in the ML pipeline that prints out the transformed data that will enter the model
class Debug(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        self.shape = X.shape
        print(self.shape)
        X_df = pd.DataFrame(X)
        print(X_df)
        # print(X_df.to_string()) # can only be print like this without running LagVars() to avoid crashing
        # what other output you want
        return X
    
# assigning X and y for the univariate naive prediction
y_naive = (final_bike_df['total_cust'].copy())
X_naive = y_naive.copy()
X_naive = pd.DataFrame(data=X_naive, columns=['total_cust'])

# getting the start time
start_time = datetime.now()

# creating y_pred
y_pred = (X_naive.shift())[1:]
# adjusting length of the actual target values
y_naive = y_naive[1:]

# get final time
end_time = datetime.now()
print('Total running time of naive predictor:', (end_time - start_time).total_seconds())

# calculating the scores for the last value method
print('RMSE:', np.sqrt(mean_squared_error(y_naive, y_pred)))
print('RMSLE:', np.sqrt(mean_squared_log_error(y_naive, y_pred)))
print('MAE:', mean_absolute_error(y_naive, y_pred))

y_log = final_bike_df[['total_cust']].copy()
y_log['total_cust'] = y_log['total_cust'].apply(lambda x: np.log1p(x+1))
y_shift = y_log.shift(1)
y_diff = (y_log - y_shift)[1:]
X_diff = X[1:]

y_diff.reset_index(drop=True, inplace=True)
X_diff.reset_index(drop=True, inplace=True)

# initializing the model which is a Random Forest model and uses default hyperparameters
model_rf_diff = RandomForestRegressor(random_state=42)

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_rf_diff)
])

pipeline_rf_diff = pipeline.fit(X_diff, y_diff)

# creating a timeseries split of the datasets
time_split = TimeSeriesSplit(n_splits=10)

# doing cross validation on the chunks of data and calculating scores
scores_rf_diff = cross_validate(pipeline_rf_diff, X_diff, y_diff, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'],
                         return_train_score=True, n_jobs=-1)

# root mean squared error
print('Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf_diff['train_neg_mean_squared_error']])/len(scores_rf_diff['train_neg_mean_squared_error']))
print('Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf_diff['test_neg_mean_squared_error']])/len(scores_rf_diff['test_neg_mean_squared_error']))

# mean absolute error
print('Average MAE train data:', 
      sum([(-1 * x) for x in scores_rf_diff['train_neg_mean_absolute_error']])/len(scores_rf_diff['train_neg_mean_absolute_error']))
print('Average MAE test data:', 
      sum([(-1 * x) for x in scores_rf_diff['test_neg_mean_absolute_error']])/len(scores_rf_diff['test_neg_mean_absolute_error']))

# getting indices for all folds from timeseriessplit
X_train_over = defaultdict()  
X_test_over = defaultdict()
y_train_over = defaultdict()
y_test_over = defaultdict()

y_test_index = []
y_train_index = []

count = 0

for train_index, test_index in time_split.split(X_diff):
    print("TRAIN:", train_index, "TEST:", test_index)
    
    y_test_index.append(test_index)
    y_train_index.append(train_index)
    
    X_train, X_test = X_diff.iloc[train_index], X_diff.iloc[test_index]
    y_train, y_test = (np.array(y_diff))[train_index], (np.array(y_diff))[test_index]
    X_train_over[count] = X_train
    X_test_over[count] = X_test
    y_train_over[count] = y_train
    y_test_over[count] = y_test
    
    count += 1
def split_predict_test(X_train, y_train, X_test, y_test, split):
    
    pipeline_part = pipeline.fit(X_train[split], y_train[split].ravel())
    prediction_test = pipeline_part.predict(X_test[split])
    #print('transformed RMSE:', np.sqrt(mean_squared_error(y_test[split], prediction_test)))
        
    # what are the predictions in absolute terms
    y_test_pred_log = prediction_test + y_log.iloc[y_test_index[split]]['total_cust']
    y_test_pred = np.exp(y_test_pred_log) - 2

    # what are the actual values in absolute terms
    y_test_true_log = y_test[split] + y_log.iloc[y_test_index[split]]
    y_test_true = np.exp(y_test_true_log) - 2 # minus 2 because we added 1 each when doing the log function earlier

    # what is the rmse, rmsle and mae
    #print('actual RMSE:', np.sqrt(mean_squared_error(y_test_true, y_test_pred)))
    #print('actual RMSLE:', np.sqrt(mean_squared_log_error(y_test_true, y_test_pred)))
    #print('actual MAE:', mean_absolute_error(y_test_true, y_test_pred))
    
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_test_true, y_test_pred))
    mae = mean_absolute_error(y_test_true, y_test_pred)
    
    list_scores = []
    list_scores.extend([rmse, rmsle, mae])
    
    return list_scores

def split_predict_train(X_train, y_train, X_test, y_test, split):
    
    pipeline_part = pipeline.fit(X_train[split], y_train[split].ravel())
    prediction_train = pipeline_part.predict(X_train[split])
    #print('transformed RMSE:', np.sqrt(mean_squared_error(y_test[split], prediction_test)))
        
    # what are the predictions in absolute terms
    y_train_pred_log = prediction_train + y_log.iloc[y_train_index[split]]['total_cust']
    y_train_pred = np.exp(y_train_pred_log) - 2

    # what are the actual values in absolute terms
    y_train_true_log = y_train[split] + y_log.iloc[y_train_index[split]]
    y_train_true = np.exp(y_train_true_log) - 2 # minus 2 because we added 1 each when doing the log function earlier

    # what is the rmse, rmsle and mae
    #print('actual RMSE:', np.sqrt(mean_squared_error(y_test_true, y_test_pred)))
    #print('actual RMSLE:', np.sqrt(mean_squared_log_error(y_test_true, y_test_pred)))
    #print('actual MAE:', mean_absolute_error(y_test_true, y_test_pred))
    
    rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_train_true, y_train_pred))
    mae = mean_absolute_error(y_train_true, y_train_pred)
    
    list_scores = []
    list_scores.extend([rmse, rmsle, mae])
    
    return list_scores
# initializing the model which is a Random Forest model and uses default hyperparameters
model_rf = RandomForestRegressor(bootstrap=False, random_state=15)

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_rf)
])

pipeline_rf = pipeline.fit(X, y)

# creating a timeseries split of the datasets
time_split = TimeSeriesSplit(n_splits=10)

# doing cross validation on the chunks of data and calculating scores
scores_rf = cross_validate(pipeline_rf, X, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# root mean squared error
print('Random Forests: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf['train_neg_mean_squared_error']])/len(scores_rf['train_neg_mean_squared_error']))
print('Random Forests: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf['test_neg_mean_squared_error']])/len(scores_rf['test_neg_mean_squared_error']))

# mean absolute error
print('Random Forests: Average MAE train data:', 
      sum([(-1 * x) for x in scores_rf['train_neg_mean_absolute_error']])/len(scores_rf['train_neg_mean_absolute_error']))
print('Random Forests: Average MAE test data:', 
      sum([(-1 * x) for x in scores_rf['test_neg_mean_absolute_error']])/len(scores_rf['test_neg_mean_absolute_error']))

# root mean squared log error
print('Random Forests: Average RMSLE train data:', 
      sum([(-1 * x) for x in scores_rf['train_neg_mean_squared_log_error']])/len(scores_rf['train_neg_mean_squared_log_error']))
print('Random Forests: Average RMSLE test data:', 
      sum([(-1 * x) for x in scores_rf['test_neg_mean_squared_log_error']])/len(scores_rf['test_neg_mean_squared_log_error']))

# number of trees in random forest
n_estimators = randint(1, 2000)
# maximum number of features included in the model
max_features = randint(1, 20)
# maximum number of levels in tree
max_depth = randint(1,10)
# minimum number of samples required to split a node
min_samples_leaf = randint(1, 10)

# create the random grid
random_grid_rf = {'model__n_estimators': n_estimators,
                   'model__max_depth': max_depth,
                   'model__min_samples_leaf': min_samples_leaf,
                   'model__max_features': max_features}

# check the start time
start_time = datetime.now()
print(start_time)

# instantiate and fit the randomizedsearch class to the random parameters
rs_rf = RandomizedSearchCV(pipeline_rf, 
                        param_distributions=random_grid_rf, 
                        scoring='neg_mean_squared_error', 
                        n_jobs=-1,
                        cv=time_split,
                        n_iter = 5,
                        verbose=10,
                        random_state=49)
rs_rf = rs_rf.fit(X, y)

# print the total running time
end_time = datetime.now()
print('Total running time:', end_time-start_time)

# Saving the best RandomForest model
pickle.dump(rs_rf.best_estimator_, open('randomforest.sav', 'wb'))

# change the keys of the best_params dictionary to allow it to be used for the final model
fix_best_params_rf = {key[7:]: val for key, val in rs_rf.best_params_.items()}

print(fix_best_params_rf)

# fit the randomforestregressor with the best_params as hyperparameters
model_rf_rs = RandomForestRegressor(**fix_best_params_rf, bootstrap=False)

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_rf_rs)
])

# fitting x and y to pipeline
pipeline_rf_rs = pipeline.fit(X, y)

# doing cross validation on the chunks of data and calculating scores
scores_rf = cross_validate(pipeline_rf_rs, X, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# root mean squared error
print('Random Forests: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf['train_neg_mean_squared_error']])/len(scores_rf['train_neg_mean_squared_error']))
print('Random Forests: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf['test_neg_mean_squared_error']])/len(scores_rf['test_neg_mean_squared_error']))

# absolute mean error
print('Random Forests: Average MAE train data:', 
      sum([(-1 * x) for x in scores_rf['train_neg_mean_absolute_error']])/len(scores_rf['train_neg_mean_absolute_error']))
print('Random Forests: Average MAE test data:', 
      sum([(-1 * x) for x in scores_rf['test_neg_mean_absolute_error']])/len(scores_rf['test_neg_mean_absolute_error']))

# root mean squared log error
print('Random Forests: Average RMSLE train data:', 
      sum([(-1 * x) for x in scores_rf['train_neg_mean_squared_log_error']])/len(scores_rf['train_neg_mean_squared_log_error']))
print('Random Forests: Average RMSLE test data:', 
      sum([(-1 * x) for x in scores_rf['test_neg_mean_squared_log_error']])/len(scores_rf['test_neg_mean_squared_log_error']))

# assign lists of parameters to be used in gridsearch
param_grid_rf = {'model__max_depth': [8, 9],
                 'model__max_features': [18, 19],
                 'model__min_samples_leaf': [5, 6],
                 'model__n_estimators': [1926, 1930]
}

# use gridsearch to search around the randomizedsearch best parameters and further improve the model
gs_rf = GridSearchCV(pipeline_rf, 
                  param_grid=param_grid_rf, 
                  scoring='neg_mean_squared_error', 
                  verbose = 10,
                  n_jobs=-1, 
                  cv=time_split)
gs_rf = gs_rf.fit(X, y)

# Saving the best RandomForest model
pickle.dump(gs_rf.best_estimator_, open('randomforest.sav', 'wb'))

# change the keys of the best_params dictionary to allow it to be used for the final model
fix_best_params_rf = {key[7:]: val for key, val in gs_rf.best_params_.items()}

print(fix_best_params_rf)

# fit the randomforestregressor with the best_params as hyperparameters
model_rf_tuned = RandomForestRegressor(**fix_best_params_rf, bootstrap=False)

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_rf_tuned)
])

# fitting x and y to pipeline
pipeline_rf_tuned = pipeline.fit(X, y)

# check the start time
start_time = datetime.now()

# doing cross validation on the chunks of data and calculating scores
scores_rf_tuned = cross_validate(pipeline_rf_tuned, X, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# print the total running time
end_time = datetime.now()
print('Total cross validation time for random forests:', (end_time-start_time).total_seconds())

# root mean squared error
print('Random Forests: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf_tuned['train_neg_mean_squared_error']])/len(scores_rf_tuned['train_neg_mean_squared_error']))
print('Random Forests: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf_tuned['test_neg_mean_squared_error']])/len(scores_rf_tuned['test_neg_mean_squared_error']))

# mean absolute error
print('Random Forests: Average MAE train data:', 
      sum([(-1 * x) for x in scores_rf_tuned['train_neg_mean_absolute_error']])/len(scores_rf_tuned['train_neg_mean_absolute_error']))
print('Random Forests: Average MAE test data:', 
      sum([(-1 * x) for x in scores_rf_tuned['test_neg_mean_absolute_error']])/len(scores_rf_tuned['test_neg_mean_absolute_error']))

# root mean squared log error
print('Random Forests: Average RMSLE train data:', 
      sum([(-1 * x) for x in scores_rf_tuned['train_neg_mean_squared_log_error']])/len(scores_rf_tuned['train_neg_mean_squared_log_error']))
print('Random Forests: Average RMSLE test data:', 
      sum([(-1 * x) for x in scores_rf_tuned['test_neg_mean_squared_log_error']])/len(scores_rf_tuned['test_neg_mean_squared_log_error']))


# plot feature importance
plt.figure(figsize=[12,9])
importances = list(pipeline_rf_tuned.steps[1][1].feature_importances_)

imp_dict = {key: val for key, val in zip(list(X.columns), importances)}
sorted_feats = sorted(imp_dict.items(), key=lambda x:x[1], reverse=True)
x_val = [x[0] for x in sorted_feats]
y_val = [x[1] for x in sorted_feats]

sb.barplot(y_val, x_val)

# get the final scores for the differenced data
split = 10
all_scores_test = []
all_scores_train = []

for i in range(split):
    scores_test = split_predict_test(X_train_over, y_train_over, X_test_over, y_test_over, i)
    all_scores_test.append(scores_test)
    scores_train = split_predict_train(X_train_over, y_train_over, X_test_over, y_test_over, i)
    all_scores_train.append(scores_train)
    
rmse_test = []
rmsle_test = []
mae_test = []
rmse_train = []
rmsle_train = []
mae_train = []

for vals in all_scores_test:
    rmse_test.append(vals[0])
    rmsle_test.append(vals[1])
    mae_test.append(vals[2])
    
for vals in all_scores_train:
    rmse_train.append(vals[0])
    rmsle_train.append(vals[1])
    mae_train.append(vals[2])
    
print('Overall Test RMSE:', sum(rmse_test)/split)
print('Overall Test RMSLE:', sum(rmsle_test)/split)
print('Overall Test MAE:', sum(mae_test)/split)

print('Overall Train RMSE:', sum(rmse_train)/split)
print('Overall Train RMSLE:', sum(rmsle_train)/split)
print('Overall Train MAE:', sum(mae_train)/split)
    
# how many features are in the dataset currently:
len(x_val)
weekday_feats = ['weekday_3', 'weekday_1', 'weekday_2', 'weekday_4', 'weekday_5', 'weekday_6']
# check how the model performance without certain features that are very unimportant
X_feat_imp = X.drop(columns=weekday_feats)

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window)]#, 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_rf_tuned)
])

# fitting x and y to pipeline
pipeline_rf_tuned = pipeline.fit(X_feat_imp, y)

# doing cross validation on the chunks of data and calculating scores
scores_rf_tuned = cross_validate(pipeline_rf_tuned, X_feat_imp, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# root mean squared error
print('Random Forests: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf_tuned['train_neg_mean_squared_error']])/len(scores_rf_tuned['train_neg_mean_squared_error']))
print('Random Forests: Average RMSLE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf_tuned['test_neg_mean_squared_error']])/len(scores_rf_tuned['test_neg_mean_squared_error']))

# mean absolute error
print('Random Forests: Average MAE train data:', 
      sum([(-1 * x) for x in scores_rf_tuned['train_neg_mean_absolute_error']])/len(scores_rf_tuned['train_neg_mean_absolute_error']))
print('Random Forests: Average MAE test data:', 
      sum([(-1 * x) for x in scores_rf_tuned['test_neg_mean_absolute_error']])/len(scores_rf_tuned['test_neg_mean_absolute_error']))

# root mean squared log error
print('Random Forests: Average RMSLE train data:', 
      sum([(-1 * x) for x in scores_rf_tuned['train_neg_mean_squared_log_error']])/len(scores_rf_tuned['train_neg_mean_squared_log_error']))
print('Random Forests: Average RMSLE test data:', 
      sum([(-1 * x) for x in scores_rf_tuned['test_neg_mean_squared_log_error']])/len(scores_rf_tuned['test_neg_mean_squared_log_error']))


# initializing AdaBoost model with default hyperparameters
model_ada = AdaBoostRegressor(random_state=42, base_estimator=DecisionTreeRegressor())

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_ada)
])

pipeline_ada = pipeline.fit(X, y)

# creating a timeseries split of the datasets
time_split = TimeSeriesSplit(n_splits=10)

# doing cross validation on the chunks of data and calculating scores
scores_ada = cross_validate(pipeline_ada, X, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# root mean squared error
print('AdaBoost: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada['train_neg_mean_squared_error']])/len(scores_ada['train_neg_mean_squared_error']))
print('AdaBoost: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada['test_neg_mean_squared_error']])/len(scores_ada['test_neg_mean_squared_error']))

# mean absolute error
print('AdaBoost: Average MAE train data:', 
      sum([(-1 * x) for x in scores_ada['train_neg_mean_absolute_error']])/len(scores_ada['train_neg_mean_absolute_error']))
print('AdaBoost: Average MAE test data:', 
      sum([(-1 * x) for x in scores_ada['test_neg_mean_absolute_error']])/len(scores_ada['test_neg_mean_absolute_error']))

# root mean squared log error
print('AdaBoost: Average RMSLE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada['train_neg_mean_squared_log_error']])/len(scores_ada['train_neg_mean_squared_log_error']))
print('AdaBoost: Average RMSLE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada['test_neg_mean_squared_log_error']])/len(scores_ada['test_neg_mean_squared_log_error']))

# number of estimators in AdaBoost model
n_estimators = randint(100, 1000)
# learning rate
learning_rate = uniform(0.001, 0.05)
# max_depth
max_depth = randint(1, 8)

# Create the random grid
random_grid_ada = {'model__n_estimators': n_estimators,
                   'model__learning_rate': learning_rate,
                   'model__base_estimator__max_depth': max_depth}

# check the start time
start_time = datetime.now()
print(start_time)

# instantiate and fit the randomizedsearch class to the random parameters
rs_ada = RandomizedSearchCV(pipeline_ada, 
                        param_distributions=random_grid_ada, 
                        scoring='neg_mean_squared_error', 
                        n_jobs=-1,
                        cv=time_split,
                        n_iter = 5,
                        verbose=10,
                        random_state=40)
rs_ada = rs_ada.fit(X, y)

# print the total running time
end_time = datetime.now()
print('Total running time:', end_time-start_time)
    
# Saving the best RandomForest model
pickle.dump(rs_ada.best_estimator_, open('adaboost.sav', 'wb'))

# change the keys of the best_params dictionary to allow it to be used for the final model
fix_best_params_ada = {key[7:]: val for key, val in rs_ada.best_params_.items()}

print(fix_best_params_ada)
max_depth = fix_best_params_ada['base_estimator__max_depth']

# fit the randomforestregressor with the best_params as hyperparameters
model_ada_tuned = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth),
                                    learning_rate=fix_best_params_ada['learning_rate'], 
                                    n_estimators=fix_best_params_ada['n_estimators'])

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_ada_tuned)
])

pipeline_ada_tuned = pipeline.fit(X, y)

# creating a timeseries split of the datasets
time_split = TimeSeriesSplit(n_splits=10)

# doing cross validation on the chunks of data and calculating scores
scores_ada_tuned = cross_validate(pipeline_ada_tuned, X, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# root mean squared error
print('AdaBoost: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada_tuned['train_neg_mean_squared_error']])/len(scores_ada_tuned['train_neg_mean_squared_error']))
print('AdaBoost: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada_tuned['test_neg_mean_squared_error']])/len(scores_ada_tuned['test_neg_mean_squared_error']))

# mean absolute error
print('AdaBoost: Average MAE train data:', 
      sum([(-1 * x) for x in scores_ada_tuned['train_neg_mean_absolute_error']])/len(scores_ada_tuned['train_neg_mean_absolute_error']))
print('AdaBoost: Average MAE test data:', 
      sum([(-1 * x) for x in scores_ada_tuned['test_neg_mean_absolute_error']])/len(scores_ada_tuned['test_neg_mean_absolute_error']))

# root mean squared log error
print('AdaBoost: Average RMSLE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada_tuned['train_neg_mean_squared_log_error']])/len(scores_ada_tuned['train_neg_mean_squared_log_error']))
print('AdaBoost: Average RMSLEtest data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada_tuned['test_neg_mean_squared_log_error']])/len(scores_ada_tuned['test_neg_mean_squared_log_error']))

# specify some values for hyperparameters around the values chosen by randomizedsearch
grid_param_ada = {'model__learning_rate': [0.003, 0.004, 0.0332],
                  'model__n_estimators': [100, 265],
                  'model__base_estimator__max_depth': [5, 6, 7]}


# use gridsearch to search around the randomizedsearch best parameters and further improve the model
gs_ada = GridSearchCV(pipeline_ada, 
                  param_grid=grid_param_ada, 
                  scoring='neg_mean_squared_error',
                  verbose = 10,
                  n_jobs=-1, 
                  cv=time_split)

gs_ada = gs_ada.fit(X, y)

# Saving the best RandomForest model
#pickle.dump(gs_ada.best_estimator_, open('adaboost.sav', 'wb'))

# change the keys of the best_params dictionary to allow it to be used for the final model
fix_best_params_ada = {key[7:]: val for key, val in gs_ada.best_params_.items()}

print(fix_best_params_ada)

max_depth = fix_best_params_ada['base_estimator__max_depth']

# fit the randomforestregressor with the best_params as hyperparameters
model_ada_tuned = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth),
                                    learning_rate=fix_best_params_ada['learning_rate'], 
                                    n_estimators=fix_best_params_ada['n_estimators'])

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_ada_tuned)
])

pipeline_ada_tuned = pipeline.fit(X, y)

# creating a timeseries split of the datasets
time_split = TimeSeriesSplit(n_splits=10)

# check the start time
start_time = datetime.now()

# doing cross validation on the chunks of data and calculating scores
scores_ada_tuned = cross_validate(pipeline_ada_tuned, X, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# print the total running time
end_time = datetime.now()
print('Total cross validation time for AdaBoost:', (end_time-start_time).total_seconds())

# root mean squared error
print('AdaBoost: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada_tuned['train_neg_mean_squared_error']])/len(scores_ada_tuned['train_neg_mean_squared_error']))
print('AdaBoost: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada_tuned['test_neg_mean_squared_error']])/len(scores_ada_tuned['test_neg_mean_squared_error']))

# mean average error
print('AdaBoost: Average MAE train data:', 
      sum([(-1 * x) for x in scores_ada_tuned['train_neg_mean_absolute_error']])/len(scores_ada_tuned['train_neg_mean_absolute_error']))
print('AdaBoost: Average MAE test data:', 
      sum([(-1 * x) for x in scores_ada_tuned['test_neg_mean_absolute_error']])/len(scores_ada_tuned['test_neg_mean_absolute_error']))

# root mean squared log error
print('AdaBoost: Average RMSLE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada_tuned['train_neg_mean_squared_log_error']])/len(scores_ada_tuned['train_neg_mean_squared_log_error']))
print('AdaBoost: Average RMSLE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada_tuned['test_neg_mean_squared_log_error']])/len(scores_ada_tuned['test_neg_mean_squared_log_error']))


# plot feature importance
plt.figure(figsize=[12,9])
importances = list(pipeline_ada_tuned.steps[1][1].feature_importances_)

imp_dict = {key: val for key, val in zip(list(X.columns), importances)}
sorted_feats = sorted(imp_dict.items(), key=lambda x:x[1], reverse=True)
x_val = [x[0] for x in sorted_feats]
y_val = [x[1] for x in sorted_feats]

#importances.sort(reverse=True)
sb.barplot(y_val, x_val)
#importances
# get the final scores for the differenced data
split = 10
all_scores_test = []
all_scores_train = []

for i in range(split):
    scores_test = split_predict_test(X_train_over, y_train_over, X_test_over, y_test_over, i)
    all_scores_test.append(scores_test)
    scores_train = split_predict_train(X_train_over, y_train_over, X_test_over, y_test_over, i)
    all_scores_train.append(scores_train)
    
rmse_test = []
rmsle_test = []
mae_test = []
rmse_train = []
rmsle_train = []
mae_train = []

for vals in all_scores_test:
    rmse_test.append(vals[0])
    rmsle_test.append(vals[1])
    mae_test.append(vals[2])
    
for vals in all_scores_train:
    rmse_train.append(vals[0])
    rmsle_train.append(vals[1])
    mae_train.append(vals[2])
    
print('Overall Test RMSE:', sum(rmse_test)/split)
print('Overall Test RMSLE:', sum(rmsle_test)/split)
print('Overall Test MAE:', sum(mae_test)/split)

print('Overall Train RMSE:', sum(rmse_train)/split)
print('Overall Train RMSLE:', sum(rmsle_train)/split)
print('Overall Train MAE:', sum(mae_train)/split)
    
# check how the model performance without certain features that are very unimportant
X_feat_imp = X.drop(columns=['weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6'])

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window)]#, 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_ada_tuned)
])

pipeline_ada_tuned = pipeline.fit(X_feat_imp, y)

# creating a timeseries split of the datasets
time_split = TimeSeriesSplit(n_splits=10)

# doing cross validation on the chunks of data and calculating scores
scores_ada_tuned = cross_validate(pipeline_ada_tuned, X_feat_imp, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# root mean squared error
print('AdaBoost: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada_tuned['train_neg_mean_squared_error']])/len(scores_ada_tuned['train_neg_mean_squared_error']))
print('AdaBoost: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_ada_tuned['test_neg_mean_squared_error']])/len(scores_ada_tuned['test_neg_mean_squared_error']))

# mean average error
print('AdaBoost: Average MAE train data:', 
      sum([(-1 * x) for x in scores_ada_tuned['train_neg_mean_absolute_error']])/len(scores_ada_tuned['train_neg_mean_absolute_error']))
print('AdaBoost: Average MAE test data:', 
      sum([(-1 * x) for x in scores_ada_tuned['test_neg_mean_absolute_error']])/len(scores_ada_tuned['test_neg_mean_absolute_error']))

# root mean squared log error
print('AdaBoost: Average RMSLE train data:', 
      sum([(-1 * x) for x in scores_ada_tuned['train_neg_mean_squared_log_error']])/len(scores_ada_tuned['train_neg_mean_squared_log_error']))
print('AdaBoost: Average RMSLE test data:', 
      sum([(-1 * x) for x in scores_ada_tuned['test_neg_mean_squared_log_error']])/len(scores_ada_tuned['test_neg_mean_squared_log_error']))



# initializing XGBoost model with default hyperparameters
model_xgb = xgb.XGBRegressor(random_state=42)

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', MinMaxScaler()),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_xgb)
])

pipeline_xgb = pipeline.fit(X, y)

# creating a timeseries split of the datasets
time_split = TimeSeriesSplit(n_splits=10)

# doing cross validation on the chunks of data and calculating scores
scores_xgb = cross_validate(pipeline_xgb, X, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# root mean squared error
print('XGBoost: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb['train_neg_mean_squared_error']])/len(scores_xgb['train_neg_mean_squared_error']))
print('XGBoost: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb['test_neg_mean_squared_error']])/len(scores_xgb['test_neg_mean_squared_error']))

# mean absolute error
print('XGBoost: Average MAE train data:', 
      sum([(-1 * x) for x in scores_xgb['train_neg_mean_absolute_error']])/len(scores_xgb['train_neg_mean_absolute_error']))
print('XGBoost: Average MAE test data:', 
      sum([(-1 * x) for x in scores_xgb['test_neg_mean_absolute_error']])/len(scores_xgb['test_neg_mean_absolute_error']))

# root mean squared log error
print('XGBoost: Average RMSLE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb['train_neg_mean_squared_log_error']])/len(scores_xgb['train_neg_mean_squared_log_error']))
print('XGBoost: Average RMSLE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb['test_neg_mean_squared_log_error']])/len(scores_xgb['test_neg_mean_squared_log_error']))

# alpha
alpha = uniform(0.2, 0.5)
# learning rate
learning_rate = uniform(0, 1)
# colsample_bytree
colsample_bytree = uniform(0, 1)
# max depth
n_estimators = randint(300, 1000)
# min child weight
min_child_weight = randint(5, 10)
# max_depth
max_depth = randint(1, 5)
# subsample
# subsample = uniform(0, 1)

# Create the random grid
random_grid_xgb = {'model__n_estimators': n_estimators,
                   'model__learning_rate': learning_rate,
                   'model__reg_alpha': alpha,
                   'model__colsample_bytree': colsample_bytree,
                   'model__min_child_weight': min_child_weight,
                   'model__max_depth': max_depth}
                   #'model__subsample': subsample}

# check the start time
start_time = datetime.now()
print(start_time)

# instantiate and fit the randomizedsearch class to the random parameters
rs_xgb = RandomizedSearchCV(pipeline_xgb, 
                        param_distributions=random_grid_xgb, 
                        scoring='neg_mean_squared_error', 
                        n_jobs=-1,
                        cv=time_split,
                        n_iter = 10,
                        verbose=10,
                        random_state=10)
rs_xgb = rs_xgb.fit(X, y)

# print the total running time
end_time = datetime.now()
print('Total running time:', end_time-start_time)
    
# Saving the best RandomForest model
pickle.dump(rs_xgb.best_estimator_, open('xgboost.sav', 'wb'))

# change the keys of the best_params dictionary to allow it to be used for the final model
fix_best_params_xgb = {key[7:]: val for key, val in rs_xgb.best_params_.items()}

print(fix_best_params_xgb)

# fit the randomforestregressor with the best_params as hyperparameters
model_xgb_rs = xgb.XGBRegressor(**fix_best_params_xgb)

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', MinMaxScaler()),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_xgb_rs)
])

pipeline_xgb_rs = pipeline.fit(X, y)

# creating a timeseries split of the datasets
time_split = TimeSeriesSplit(n_splits=10)

# doing cross validation on the chunks of data and calculating scores
scores_xgb_rs = cross_validate(pipeline_xgb_rs, X, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# root mean squared error
print('XGBoost: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_rs['train_neg_mean_squared_error']])/len(scores_xgb_rs['train_neg_mean_squared_error']))
print('XGBoost: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_rs['test_neg_mean_squared_error']])/len(scores_xgb_rs['test_neg_mean_squared_error']))

# mean absolute error
print('XGBoost: Average MAE train data:', 
      sum([(-1 * x) for x in scores_xgb_rs['train_neg_mean_absolute_error']])/len(scores_xgb_rs['train_neg_mean_absolute_error']))
print('XGBoost: Average MAE test data:', 
      sum([(-1 * x) for x in scores_xgb_rs['test_neg_mean_absolute_error']])/len(scores_xgb_rs['test_neg_mean_absolute_error']))

# root mean squared log error
print('XGBoost: Average RMSLE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_rs['train_neg_mean_squared_log_error']])/len(scores_xgb_rs['train_neg_mean_squared_log_error']))
print('XGBoost: Average RMSLE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_rs['test_neg_mean_squared_log_error']])/len(scores_xgb_rs['test_neg_mean_squared_log_error']))


# specify some values for hyperparameters around the values chosen by randomizedsearch
grid_param_xgb = {'model__n_estimators': [664, 650],
                   'model__learning_rate': [0.35, 0.04],
                   'model__alpha': [0.636, 0.6],
                   'model__colsample_bytree': [0.85, 0.9],
                   'model__min_child_weight': [7, 8],
                   'model__max_depth': [1, 2]}
                   #'model__subsample': [0.98, 0.90]}

# use gridsearch to search around the randomizedsearch best parameters and further improve the model
gs_xgb = GridSearchCV(pipeline_xgb, 
                  param_grid=grid_param_xgb, 
                  scoring='neg_mean_squared_error', 
                  verbose = 10,
                  n_jobs=-1, 
                  cv=time_split)

gs_xgb = gs_xgb.fit(X, y)

# Saving the best XGB model
pickle.dump(gs_xgb.best_estimator_, open('xgboost.sav', 'wb'))

# change the keys of the best_params dictionary to allow it to be used for the final model
fix_best_params_xgb = {key[7:]: val for key, val in gs_xgb.best_params_.items()}

print(fix_best_params_xgb)

# fit the randomforestregressor with the best_params as hyperparameters
model_xgb_tuned = xgb.XGBRegressor(**fix_best_params_xgb)

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', MinMaxScaler()),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_xgb_tuned)
])

pipeline_xgb_tuned = pipeline.fit(X, y)

# creating a timeseries split of the datasets
time_split = TimeSeriesSplit(n_splits=10)

# check the start time
start_time = datetime.now()

# doing cross validation on the chunks of data and calculating scores
scores_xgb_tuned = cross_validate(pipeline_xgb_tuned, X, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# print the total running time
end_time = datetime.now()
print('Total cross validation time for XGBBoost:', (end_time-start_time).total_seconds())

# root mean squared error
print('XGBoost: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_tuned['train_neg_mean_squared_error']])/len(scores_xgb_tuned['train_neg_mean_squared_error']))
print('XGBoost: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_tuned['test_neg_mean_squared_error']])/len(scores_xgb_tuned['test_neg_mean_squared_error']))

# mean absolute error
print('XGBoost: Average MAE train data:', 
      sum([(-1 * x) for x in scores_xgb_tuned['train_neg_mean_absolute_error']])/len(scores_xgb_tuned['train_neg_mean_absolute_error']))
print('XGBoost: Average MAE test data:', 
      sum([(-1 * x) for x in scores_xgb_tuned['test_neg_mean_absolute_error']])/len(scores_xgb_tuned['test_neg_mean_absolute_error']))

# root mean squared log error
print('XGBoost: Average RMSLE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_tuned['train_neg_mean_squared_log_error']])/len(scores_xgb_tuned['train_neg_mean_squared_log_error']))
print('XGBoost: Average RMSLE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_tuned['test_neg_mean_squared_log_error']])/len(scores_xgb_tuned['test_neg_mean_squared_log_error']))

# map the feature importances to the feature names
xgb_feats = X.columns
xgb_feats_dict = defaultdict()
count = 0

for item in xgb_feats:
    xgb_feats_dict['f'+str(count)] = item
    count += 1
    
# Plot feature importance
fig, ax = plt.subplots(figsize=(20, 15))
xgb.plot_importance(model_xgb_tuned, importance_type='weight', ax=ax)

# rename the y axis labels to the actual feature names
locs, labels = plt.yticks()

new_names = []
for item in labels:
    for key, name in xgb_feats_dict.items():
        if key == item.get_text():
            new_names.append(name)

ax.set_yticklabels(new_names);
            
plt.show()

# get the final scores for the differenced data
split = 10
all_scores_test = []
all_scores_train = []

for i in range(split):
    scores_test = split_predict_test(X_train_over, y_train_over, X_test_over, y_test_over, i)
    all_scores_test.append(scores_test)
    scores_train = split_predict_train(X_train_over, y_train_over, X_test_over, y_test_over, i)
    all_scores_train.append(scores_train)
    
rmse_test = []
rmsle_test = []
mae_test = []
rmse_train = []
rmsle_train = []
mae_train = []

for vals in all_scores_test:
    rmse_test.append(vals[0])
    rmsle_test.append(vals[1])
    mae_test.append(vals[2])
    
for vals in all_scores_train:
    rmse_train.append(vals[0])
    rmsle_train.append(vals[1])
    mae_train.append(vals[2])
    
print('Overall Test RMSE:', sum(rmse_test)/split)
print('Overall Test RMSLE:', sum(rmsle_test)/split)
print('Overall Test MAE:', sum(mae_test)/split)

print('Overall Train RMSE:', sum(rmse_train)/split)
print('Overall Train RMSLE:', sum(rmsle_train)/split)
print('Overall Train MAE:', sum(mae_train)/split)
    
# check how the model performance without certain features that are very unimportant
X_feat_imp = X.drop(columns=['weekday_1', 'weekday_3', 'weekday_4', 'weekday_6',
                             'hot', 'cold', 'cool', 'warm'])

# creating and fitting the ML pipeline
trend_features = ['total_cust_mean'+str(window), 'total_cust_std'+str(window), 'total_cust_t-1']

preprocessor = ColumnTransformer([
    ('trend_diff', FunctionTransformer(np.log1p, validate=False), trend_features),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', MinMaxScaler()),
    #('debug', Debug()) # I have commented this out because it will hinder the execution of this pipeline
    ('model', model_xgb_tuned)
])

pipeline_xgb_tuned = pipeline.fit(X_feat_imp, y)

# creating a timeseries split of the datasets
time_split = TimeSeriesSplit(n_splits=10)

# doing cross validation on the chunks of data and calculating scores
scores_xgb_tuned = cross_validate(pipeline_xgb_tuned, X_feat_imp, y, cv=time_split,
                         scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                         return_train_score=True, n_jobs=-1)

# root mean squared error
print('XGBoost: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_tuned['train_neg_mean_squared_error']])/len(scores_xgb_tuned['train_neg_mean_squared_error']))
print('XGBoost: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_tuned['test_neg_mean_squared_error']])/len(scores_xgb_tuned['test_neg_mean_squared_error']))

# mean absolute error
print('XGBoost: Average MAE train data:', 
      sum([(-1 * x) for x in scores_xgb_tuned['train_neg_mean_absolute_error']])/len(scores_xgb_tuned['train_neg_mean_absolute_error']))
print('XGBoost: Average MAE test data:', 
      sum([(-1 * x) for x in scores_xgb_tuned['test_neg_mean_absolute_error']])/len(scores_xgb_tuned['test_neg_mean_absolute_error']))

# root mean squared log error
print('XGBoost: Average RMSLE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_tuned['train_neg_mean_squared_log_error']])/len(scores_xgb_tuned['train_neg_mean_squared_log_error']))
print('XGBoost: Average RMSLE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_xgb_tuned['test_neg_mean_squared_log_error']])/len(scores_xgb_tuned['test_neg_mean_squared_log_error']))



