!pip install pmdarima
import pandas as pd

import numpy as np

%matplotlib inline



from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from pmdarima import auto_arima



import seaborn as sns

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from scipy import stats

from scipy.stats import skew



import warnings

warnings.filterwarnings('ignore')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv', index_col='datetime', parse_dates=True)

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv', index_col='datetime', parse_dates=True)
train.head()
test.head()
Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
train.shape
train_without_outliers = train[~((train < (Q1 - 1.5*IQR)) | (train > (Q3 + 1.5*IQR))).any(axis=1)]
train_without_outliers.shape
def wind(cols):

    windspeed = cols[0]

    season = cols[1]

    

    if(windspeed == 0):

        if(season == 1):

            return 14

        elif(season == 2):

            return 14

        else:

            return 13

    else:

        return windspeed
train_without_outliers['wind'] = train_without_outliers[['windspeed', 'season']].apply(wind, axis=1)

test['wind'] = test[['windspeed', 'season']].apply(wind, axis=1)
data = train_without_outliers.append(test)
data.tail(5)
data.dtypes
data.shape
data.isnull().sum()
data['count'][0:250].plot(figsize=(16, 5))
data['season'] = data['season'].replace({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter' })

data['weather'] = data['weather'].replace({1: 'Clear, Few clouds, Partly cloudy, Partly cloudy',

                                        2: 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',

                                        3: 'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',

                                        4: 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog' })
for i in data.columns:

    print(i)

    print(data[i].value_counts())

    print('\n')
corr = data.corr()

corr['count'][:-1]
for i in (data.select_dtypes(include ='object').columns):

    if(i != 'count'):

        data_crosstab = pd.crosstab(data[i], data['count'], margins = False)

        stat, p, dof, expected = stats.chi2_contingency(data_crosstab)

        prob=0.95

        alpha = 1.0 - prob

        if p <= alpha:

            print(i, ' : Dependent (reject H0)')

        else:

            print(i, ' : Independent (fail to reject H0)')
data.drop('casual', axis=1, inplace=True)

data.drop('registered', axis=1, inplace=True)
corr_matrix = data.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

data = data.drop(data[to_drop], axis=1)
data.head()
import matplotlib.pyplot as plt

for i in (data.skew().index):

    plt.figure(i)

    sns.distplot(data[i], kde_kws={'bw':0.1})
plt.scatter(data.index, data['count'])
def fixing_skewness(df):

    numeric_feats = df.dtypes[df.dtypes != object].index

    

    skew_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

    

    high_skew = skew_feats[abs(skew_feats) > 0.5].index

    

    for i in high_skew:

        df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))

#         print(i)

        

fixing_skewness(data)
data.head()
data.dtypes
data.shape
def overfit_reducer(df):

    overfit = []

    for i in df.columns:

        count = df[i].value_counts()

        zero_index_value = count.iloc[0]

        

        if (((zero_index_value / len(df)) * 100) > 99.94):

            overfit.append(i)

            

    overfit = list(overfit)

    return overfit
#Finding the list of overfitted features using above user-defined function

overfitted_features = overfit_reducer(data)

#Dropping the overfitted columns from the final dataframes

data.drop(overfitted_features, axis=1, inplace=True)
data.shape
data.dtypes
data1 = data.dropna()
data1.shape
data1.tail()
data1.head()
data1['count'] = data1['count'].astype('int64')
data1.head()
title = 'Count of bikes rented'

data1['count'].plot(figsize=(16,5), legend=True, title=title)
for i in (data1.dtypes[data1.dtypes == 'object'].index):

    plt.figure(i)

    sns.boxplot(x=i, y='count', data=data1)
for i in (data1.dtypes[data1.dtypes == 'int64'].index):

    if(i!='count'):

        plt.figure(i)

        sns.scatterplot(x=i, y='count', data=data1)
title='Count of bikes rented'



ax = data1['count'][:1000].plot(figsize=(16,5),title=title)

ax.autoscale(axis='x',tight=True)

for x in data1[:1000].query('holiday==1').index:       

    ax.axvline(x=x, color='k', alpha = 0.3);  
title='Count of bikes rented'



ax = data1['count'][:1000].plot(figsize=(16,5),title=title)

ax.autoscale(axis='x',tight=True)

for x in data1[:1000].query('holiday==0').index:       

    ax.axvline(x=x, color='k', alpha = 0.3);  
title='Count of bikes rented'



ax = data1['count'][:1000].plot(figsize=(16,5),title=title)

ax.autoscale(axis='x',tight=True)

for x in data1[:1000].query('workingday==1').index:       

    ax.axvline(x=x, color='k', alpha = 0.3);  
title='Count of bikes rented'



ax = data1['count'][:1000].plot(figsize=(16,5),title=title)

ax.autoscale(axis='x',tight=True)

for x in data1[:1000].query('workingday==0').index:       

    ax.axvline(x=x, color='k', alpha = 0.3);  
result = seasonal_decompose(data1['count'], model='multiplicative', period=24)

result.plot();
from statsmodels.tsa.stattools import adfuller



def adf_test(series,title=''):

    """

    Pass in a time series and an optional title, returns an ADF report

    """

    print(f'Augmented Dickey-Fuller Test: {title}')

    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    

    labels = ['ADF test statistic','p-value','# lags used','# observations']

    out = pd.Series(result[0:4],index=labels)



    for key,val in result[4].items():

        out[f'critical value ({key})']=val

        

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    

    if result[1] <= 0.05:

        print("Strong evidence against the null hypothesis")

        print("Reject the null hypothesis")

        print("Data has no unit root and is stationary")

    else:

        print("Weak evidence against the null hypothesis")

        print("Fail to reject the null hypothesis")

        print("Data has a unit root and is non-stationary")
adf_test(data1['count'])
# For SARIMA Orders we set seasonal=True and pass in an m value

# auto_arima(data1['count'],seasonal=True,m=24, trace=True, n_jos=-1).summary()
data_dummies = pd.get_dummies(data, drop_first=True)

data_dummies.columns
model = SARIMAX(data1['count'], exog=data_dummies[:7026][['holiday', 'workingday', 'temp','humidity', 'wind', 

                                           'season_spring', 'season_summer', 'season_winter',

                                           'weather_Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog',

                                           'weather_Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',

                                           'weather_Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist']],

                                            order=(2, 1, 2), seasonal_order=(1, 0, 1, 24), enforce_invertibility=False)
results=model.fit()
results.summary()
start = len(train_without_outliers)

end = len(train_without_outliers) + len(test) - 1

# exog_forecast = data[10886:][['holiday', 'workingday', 'temp','humidity', 'windspeed']]

fcast = results.predict(start=start, end=end, exog=data_dummies[7026:][['holiday', 'workingday', 'temp','humidity', 'wind', 

                                           'season_spring', 'season_summer', 'season_winter',

                                           'weather_Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog',

                                           'weather_Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',

                                           'weather_Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist']]).rename('SARIMAX(1, 1, 1)x(2, 0, [1, 2], 2) Forecast')
fcast = fcast.astype('int64')

fcast = fcast.reset_index(drop=True)
fcast.index
fcast.max()
fcast.min()
submission_df = pd.DataFrame()

submission_df['datetime'] = test.index
submission_df['count'] = fcast
submission_df.head()
submission_df.to_csv('/kaggle/working/submission.csv', index=False)