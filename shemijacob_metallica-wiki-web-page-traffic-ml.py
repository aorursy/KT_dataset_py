# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

import statsmodels.api as sm



import matplotlib.pyplot as plt # plotting

%matplotlib inline

import seaborn as sns # plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base_url = '/kaggle/input/web-traffic-time-series-forecasting/'



train_1 = pd.read_csv(base_url+'train_1.csv')

# train_2 = pd.read_csv(base_url+'train_2.csv')
train_1.shape
train_1.head()
trainT = train_1.drop('Page', axis=1).T

trainT.columns = train_1.Page.values

trainT.head()
metallica = pd.DataFrame(trainT['Metallica_es.wikipedia.org_all-access_all-agents'])

metallica.head()
print (metallica.shape)
print (metallica.isnull().sum())
plt.figure(figsize=(24, 12))

metallica.plot();
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):



    """

        series - dataframe with timeseries

        window - rolling window size 

        plot_intervals - show confidence intervals

        plot_anomalies - show anomalies 



    """

    # Calculate and plot rolling mean

    rolling_mean = series.rolling(window=window).mean()

    

    plt.figure(figsize=(15,5))

    plt.title("Moving average\n window size = {}".format(window))

    plt.plot(rolling_mean, "g", label="Rolling mean trend")



    # Plot confidence intervals for smoothed values

    if plot_intervals:

        mae = mean_absolute_error(series[window:], rolling_mean[window:])

        deviation = np.std(series[window:] - rolling_mean[window:])

        lower_bond = rolling_mean - (mae + scale * deviation)

        upper_bond = rolling_mean + (mae + scale * deviation)

        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")

        plt.plot(lower_bond, "r--")

        

        # Having the intervals, find abnormal values

        if plot_anomalies:

            anomalies = pd.DataFrame(index=series.index, columns=series.columns)

            anomalies[series<lower_bond] = series[series<lower_bond]

            anomalies[series>upper_bond] = series[series>upper_bond]

            plt.plot(anomalies, "ro", markersize=10)

    

    # Plot original series values

    plt.plot(series[window:], label="Actual values")

    plt.legend(loc="upper left")

    plt.grid(True)
plotMovingAverage(metallica, 14)
metallica1 = pd.DataFrame(metallica.copy())

metallica1.columns = ["y"]
for i in range(6, 25):

    metallica1["lag_{}".format(i)] = metallica1.y.shift(i)

    

metallica1.tail(3)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import TimeSeriesSplit 



# for time-series cross-validation set 5 folds 

tscv = TimeSeriesSplit(n_splits=5)
# Perform train-test split with respect to time series structure



def timeseries_train_test_split(X, y, test_size):

       

    # get the index after which test set starts

    test_index = int(len(X)*(1-test_size))

    

    X_train = X.iloc[:test_index]

    y_train = y.iloc[:test_index]

    X_test = X.iloc[test_index:]

    y_test = y.iloc[test_index:]

    

    return X_train, X_test, y_train, y_test
y = metallica1.dropna().y

X = metallica1.dropna().drop(['y'], axis=1)



# apply the function and reserve 30% of data for testing

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
lr = LinearRegression()

lr.fit(X_train, y_train)
def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):

   

    prediction = model.predict(X_test)

    

    plt.figure(figsize=(15, 7))

    plt.plot(prediction, "g", label="prediction", linewidth=2.0)

    plt.plot(y_test.values, label="actual", linewidth=2.0)
plotModelResults(lr)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



lr = LinearRegression()

lr.fit(X_train_scaled, y_train)



plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled)
from sklearn.linear_model import LassoCV, RidgeCV



ridge = RidgeCV(cv=tscv)

ridge.fit(X_train_scaled, y_train)



plotModelResults(ridge, 

                 X_train=X_train_scaled, 

                 X_test=X_test_scaled)
lasso = LassoCV(cv=tscv)

lasso.fit(X_train_scaled, y_train)



plotModelResults(lasso, 

                 X_train=X_train_scaled, 

                 X_test=X_test_scaled)             
from xgboost import XGBRegressor 



xgb = XGBRegressor()

xgb.fit(X_train_scaled, y_train);
plotModelResults(xgb, 

                 X_train=X_train_scaled, 

                 X_test=X_test_scaled)