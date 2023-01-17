# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base_url = '/kaggle/input/competitive-data-science-predict-future-sales/'

sales_train = pd.read_csv(base_url+'sales_train.csv', parse_dates=['date'])

items = pd.read_csv(base_url+'items.csv')

categories = pd.read_csv(base_url+'item_categories.csv')

shops = pd.read_csv(base_url+'shops.csv')
sales_train.info()
sales_train.head()
sales_train.describe()
sales_train.isnull().sum()
items.head()
items.info()
items.isnull().sum()
items.describe()
categories.isnull().sum()
shops.head()
shops.isnull().sum()
aux_0 = sales_train.merge(items, how='inner', on='item_id')

aux_1 = aux_0.merge(categories, how='inner', on='item_category_id')

df = aux_1.merge(shops, how='inner', on='shop_id')

df.head()
def clean_negatives(item_cnt_day):

    return abs(item_cnt_day)



df['total_sales'] = df.item_cnt_day.apply(clean_negatives)
df_daily = pd.DataFrame(df.groupby('date', as_index=False).total_sales.sum())
df_daily.columns
df_daily.head()
plt.figure(figsize=(20,14))

plt.plot(df_daily.date, df_daily.total_sales)
#df.set_index('date', inplace=True)

weekly_df = df.resample('W').sum()
weekly_df.columns
plt.figure(figsize=(20, 14))

plt.plot_date(weekly_df.index, weekly_df.total_sales, '.-')
weekly_df_prophet = weekly_df[['total_sales']]
weekly_df_prophet.reset_index(inplace=True)

weekly_df_prophet.columns
weekly_df_prophet.rename(columns={'date':'ds', 'total_sales':'y'}, inplace=True)

weekly_df_prophet.head()
import fbprophet

from fbprophet import Prophet
prediction_size = 12

weekly_prophet_model = Prophet()

weekly_prophet_model.fit(weekly_df_prophet)

weekly_future = weekly_prophet_model.make_future_dataframe(periods=prediction_size)

weekly_forecast = weekly_prophet_model.predict(weekly_future)
weekly_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
weekly_prophet_model.plot(weekly_forecast,

              uncertainty=True);
weekly_prophet_model.plot_components(weekly_forecast);
print('hello dh')
# Import everything from above

from sklearn.metrics import mean_squared_error

import math



# Function for mean_absolute_percentage_error

def rmse(y_actual, y_predicted):

    return math.sqrt(mean_squared_error(y_actual, y_predicted))
# Function to plot ACF and PACF Dickey-Fuller test

from dateutil.relativedelta import relativedelta # working with dates with style

from scipy.optimize import minimize              # for function minimization



import statsmodels.formula.api as smf            # statistics and econometrics

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs



from itertools import product

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):

    """

        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        

        y - timeseries

        lags - how many lags to include in ACF, PACF calculation

    """

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

        

    with plt.style.context(style):    

        fig = plt.figure(figsize=figsize)

        layout = (2, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1, 0))

        pacf_ax = plt.subplot2grid(layout, (1, 1))

        

        y.plot(ax=ts_ax)

        p_value = sm.tsa.stattools.adfuller(y)[1]

        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)

        plt.tight_layout()
tsplot(weekly_df.total_sales.values, lags=52)
for i in range(1, 53):

    weekly_df["lag_{}".format(i)] = weekly_df.total_sales.shift(i)
weekly_df.columns
weekly_df.reset_index(inplace=True)
weekly_df['week_number'] = weekly_df.date.dt.week
weekly_df['month'] = weekly_df.date.dt.month
def weeks_to_xmas(week_number, xmas_week_number=52):

    """

    Calculates the log of the difference of weeks to xmas.

    Interested in log and not absolute value as I want the 

    magnitude of distance and not distance itself

    I: week_number int, xmas_week_number int

    O: int

    """

    return np.log1p(xmas_week_number - week_number)
weekly_df['log_weeks_to_xmas'] = weekly_df.week_number.apply(weeks_to_xmas)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import TimeSeriesSplit 



#weekly_df.set_index('date', inplace=True)

# for time-series cross-validation set 5 folds 

tscv = TimeSeriesSplit(n_splits=8)



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

del weekly_df['item_cnt_day']

y = weekly_df.dropna().total_sales

X = weekly_df.dropna().drop(['total_sales'], axis=1)



# reserve 30% of data for testing

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
# machine learning in two lines



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



lr = LinearRegression()

lr.fit(X_train_scaled, y_train)
def plotModelResults(model, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=False, plot_anomalies=False):

    """

        Plots modelled vs fact values, prediction intervals and anomalies

    

    """

    

    prediction = model.predict(X_test)

    

    plt.figure(figsize=(15, 7))

    plt.plot(prediction, "g", label="prediction", linewidth=2.0)

    plt.plot(y_test.values, label="actual", linewidth=2.0)

    

    if plot_intervals:

        cv = cross_val_score(model, X_train, y_train, 

                                    cv=tscv, 

                                    scoring="neg_mean_absolute_error")

        mae = cv.mean() * (-1)

        deviation = cv.std()

        

        scale = 1.96

        lower = prediction - (mae + scale * deviation)

        upper = prediction + (mae + scale * deviation)

        

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)

        plt.plot(upper, "r--", alpha=0.5)

        

        if plot_anomalies:

            anomalies = np.array([np.NaN]*len(y_test))

            anomalies[y_test<lower] = y_test[y_test<lower]

            anomalies[y_test>upper] = y_test[y_test>upper]

            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

    

    error = rmse(y_test, prediction)

    plt.title("RMSE {0:.2f}".format(error))

    plt.legend(loc="best")

    plt.tight_layout()

    plt.grid(True);

    

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
plotModelResults(lr, plot_intervals=True)

plotCoefficients(lr)
weekly_df.item_category_id.nunique()
weekly_df.shop_id.nunique()
columns = ['item_category_id','item_id','shop_id']

for column in columns:

    del weekly_df[column]
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV



#rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train, njobs=-1)

#rf_predict = rf.predict(X_test_scaled)



parameters = {'max_features': [4, 7, 10], 

              'min_samples_leaf': [1, 3, 5, 7], 

              'max_depth': [5, 10, 15, 20]}

rfc = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=tscv, verbose=1)

gcv.fit(X_train_scaled, y_train)
gcv.best_params_, gcv.best_score_
rf_model = RandomForestRegressor(n_estimators=100,

                                 max_depth=5,

                                 max_features=10,

                                 min_samples_leaf=3,

                                 random_state=42, 

                                 n_jobs=-1)

rf_model.fit(X_train_scaled, y_train)
plotModelResults(rf_model, plot_intervals=False)
weekly_df.head()
first_diff = [row[2] - row[3] for row in X_train_scaled]
len(X_train_scaled)
len(first_diff)