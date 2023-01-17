# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Import libraries

import numpy as np

import pandas as pd

import math as math

import geopy.distance

import matplotlib.pyplot as plt

import matplotlib.colors as colors

from scipy import stats

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def clean_data(dataframe, drop_columns, percentage_clip=2.5):

    

    """

    

    This function cleans the input dataframe. This cleaning process includes removal of 

    unneccesary/redundant columns, removal or rows with empty fields, and removal of 

    statistical outliers. Here, I define statistical outliers as anything beyond the 

    percentiles set by the input parameter "percentage_clip". I will only clean input

    data this way. Any features introduced in the latter analysis, i.e., during feature 

    engineering, will only be cut based on physical reasoning and *not* on statistical 

    anomalies.

    

    Inputs:

        

        - dataframe:       Pandas dataframe structure. The input will be copied, and the 

                           copy will be edited before output.

        

        - drop_columns:    Columns present in input dataframe to be dropped. Should be 

                           a list of column names.

    

        - percentage_clip: Data beyond the value of this perccentile (in each feature 

                           column) will be removed, on either side of its distribution.

                           A standard 3-sigma for a gaussian distribution would imply

                           that 99.87% of data should be kept, and hence 0.13% of the 

                           extreme values (0.065% on either side) in each variable 

                           should be removed. The final value of 2.5% on either side

                           was decided on iteratively, as it allowed a relatively good

                           linear regression analysis (see below).



    """

    

    # Copy input structure

    data = dataframe.copy()

    

    # Drop columns

    data.drop(drop_columns, axis='columns', inplace=True)



    # Remove rows with missing values

    data.dropna(inplace=True)

    

    # Remove caps with zero passenger counts

    data = data[(data['passenger_count'] > 0)]

    

    # Remove statistical outliers

    qlow  = data.quantile(percentage_clip / 100.)        # Division to convert % to frac.

    qhigh = data.quantile(1. - (percentage_clip / 100.)) # Mirrored quantile

    distributions = [xx for xx in data.columns if data[xx].dtype!='datetime64[ns]']



    for ii in distributions:

        data = data[(data[ii] >= qlow[ii]) & (data[ii] <= qhigh[ii])]

        

    return data
def feature_engineering(dataframe):

    

    """

    Construct new features based on clean data.

    

    Inputs:

        

        - dataframe:  Pandas dataframe structure. New features will be added to a copy 

                      of this structure.

    

    Added Features:

        

        - dr_deg:     Calculate coordinate separation based on gps initial- and final 

                      positions, read from input dataframe as 'pickup_longitude', 

                      'pickup_latitude', 'dropoff_longitude' and 'dropoff_latitude'.

                      Note that this feature is a proxy, and only reflects the shortest

                      linear path between initial and end point. It is *NOT* equivalent to

                      the physical, travelled distance on roads. In addition, for large

                      distances, one should account for Earth's curvature. However, as we

                      are dealing with taxt-fare prices within a city, curvature is

                      negligable. In addition, this effect is expected to minor relative

                      to the (ignored) perturbations from infrastructure, road-planning and

                      road choices. Hence, this justifies its removal.

                      

        - year:       Extract year information from input dataframe's datetime column. We 

                      expect a dependence based on inflation and raised taxi prices.

        

        - day:        Extracted same as year. Expect that taxi fare price may vary with the 

                      day of the week. E.g. between work days and weekend.

        

        - hour:       Extracted same as year. Expect short-term temporal variability in price 

                      to emerge from standard work-hours, evenings, and over-pay/night-time.

    

    """

    

    # Feature to approximate distance traveled

    

    # Instantiate empty dataframe and populate it with gps positions

    df_tmp = pd.DataFrame()

    df_tmp['pickup_position']=list(zip(dataframe['pickup_latitude'],dataframe['pickup_longitude']))

    df_tmp['dropoff_position'] =list(zip(dataframe['dropoff_latitude'],dataframe['dropoff_longitude']))

    

    # Copy input structure

    data = dataframe.copy()

    

    # Add feature dr to approximate distance travelled. Calculated from absolute difference in lat. & long.

    dy                 = (data['dropoff_latitude'] - data['pickup_latitude']).abs()

    dx                 = (data['dropoff_longitude'] - data['pickup_longitude']).abs()

    data['dr_deg']     = np.sqrt(dx**2 + dy**2)

        

    # Add features to model long- and short-scale temporal variations

    data['year'] = data['pickup_datetime'].dt.year

    data['day']  = data['pickup_datetime'].dt.dayofweek

    data['hour'] = data['pickup_datetime'].dt.hour

    

    # Ensure taxi drives a positive distance greater than zero

    data = data[data['dr_deg'] > 0]

    

    return data
#  --------- User inputs



# Set paths

path_train = '../input/train.csv'

path_test  = '../input/test.csv'



# Drop 'key' column: it is just a dummy-variable, similar to 'pickup_datetime'

drop_columns = ['key']



# Optional outputs, see end of notebook

tune_hyperparams             = False  # Allows me to optimize random forest (RF) model parameters

show_parameter_distributions = True   # Distributions of each parameter

show_feature_rankings        = True   # Calculates the feature importance, based on RF 



# Import raw data

df = pd.read_csv(path_train, nrows=int(5E5), header=0, parse_dates=['pickup_datetime'])
# Clean data

df_c = clean_data(df, drop_columns)
# Add features

df_cf = feature_engineering(df_c)
# Set up parameters to fit cleaned data

y_c = df_c['fare_amount']

X_c = df_c.drop(['fare_amount', 'pickup_datetime'], axis='columns')

X_c_train, X_c_valid, y_c_train, y_c_valid = train_test_split(X_c, y_c, random_state=47, test_size = int(len(y_c)/3.))
# Set up parameters to fit cleaned + featured data

y_cf = df_cf['fare_amount']

X_cf = df_cf.drop(['fare_amount', 'pickup_datetime'], axis='columns')

X_cf_train, X_cf_valid, y_cf_train, y_cf_valid = train_test_split(X_cf, y_cf, random_state=47, test_size = int(len(y_cf)/3.))
# Make linear regression models

lr_c = LinearRegression()

lr_c.fit(X_c_train, y_c_train)

lr_c_prediction = lr_c.predict(X_c_valid)



lr_cf = LinearRegression()

lr_cf.fit(X_cf_train, y_cf_train)

lr_cf_prediction = lr_cf.predict(X_cf_valid)
# Make random forest models

rf_c = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5)

rf_c.fit(X_c_train, y_c_train)

rf_c_prediction = rf_c.predict(X_c_valid)



rf_cf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5)

rf_cf.fit(X_cf_train, y_cf_train)

rf_cf_prediction = rf_cf.predict(X_cf_valid)
# --------------- Test how well we recover known prices



# Define figure dimensions

plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')

plt.rcParams['figure.figsize'] = [10, 10]



# Evaluate a gaussian kernel density estimate (kde) on data, following python documentation on:

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

nbins     = int(50)

nlevels   = int(5E2)



# --------------- LINEAR REGRESSION

# On Cleaned data

plt.figure(1)

xx,yy     = np.mgrid[y_c_valid.min():y_c_valid.max():nbins*1j, lr_c_prediction.min():lr_c_prediction.max():nbins*1j]

positions = np.vstack([xx.ravel(), yy.ravel()])

values    = np.vstack([y_c_valid, lr_c_prediction])

kernel    = stats.gaussian_kde(values)

zz        = np.reshape(kernel(positions).T, xx.shape)

plt.contour(xx, yy, zz, nlevels)

plt.scatter(y_c_valid, lr_c_prediction, s=1, label='LR_C')

plt.xlim(0, 80)

plt.ylim(0, 80)

plt.plot([0, 80], [0, 80], ls="--", c=".3")

plt.xlabel("Known Prices (validation data)")

plt.ylabel("Predicted Prices (validation data)")



# Make sure to write associated scores and test statistics

score_lr_c_train = lr_c.score(X_c_train, y_c_train)

plt.text(50., 40., r'Score: %5.2f (training)' %score_lr_c_train, fontsize=15.)

score_lr_c_valid = lr_c.score(X_c_valid, y_c_valid)

plt.text(50., 35., r'Score: %5.2f (validation)' %score_lr_c_valid, fontsize=15.)

RMSE_lr_c_train = np.sqrt(mean_squared_error(y_c_train, lr_c.predict(X_c_train)))

plt.text(50., 25., r'RMSE: %5.2f (training)' %RMSE_lr_c_train, fontsize=15.)

RMSE_lr_c_valid = np.sqrt(mean_squared_error(y_c_valid, lr_c.predict(X_c_valid)))

plt.text(50., 20., r'RMSE: %5.2f (validation)' %RMSE_lr_c_valid, fontsize=15.)



plt.legend()



# On Cleaned + featured data

plt.figure(2)

xx,yy     = np.mgrid[y_cf_valid.min():y_cf_valid.max():nbins*1j, lr_cf_prediction.min():lr_cf_prediction.max():nbins*1j]

positions = np.vstack([xx.ravel(), yy.ravel()])

values    = np.vstack([y_cf_valid, lr_cf_prediction])

kernel    = stats.gaussian_kde(values)

zz        = np.reshape(kernel(positions).T, xx.shape)

plt.contour(xx, yy, zz, nlevels)

plt.scatter(y_cf_valid, lr_cf_prediction, s=1, label='LR_CF')

plt.xlim(0, 80)

plt.ylim(0, 80)

plt.plot([0, 80], [0, 80], ls="--", c=".3")

plt.xlabel("Known Prices (validation data)")

plt.ylabel("Predicted Prices (validation data)")



# Make sure to write associated scores and test statistics

score_lr_cf_train = lr_cf.score(X_cf_train, y_cf_train)

plt.text(50., 40., r'Score: %5.2f (training)' %score_lr_cf_train, fontsize=15.)

score_lr_cf_valid = lr_cf.score(X_cf_valid, y_cf_valid)

plt.text(50., 35., r'Score: %5.2f (validation)' %score_lr_cf_valid, fontsize=15.)

RMSE_lr_cf_train = np.sqrt(mean_squared_error(y_cf_train, lr_cf.predict(X_cf_train)))

plt.text(50., 25., r'RMSE: %5.2f (training)' %RMSE_lr_cf_train, fontsize=15.)

RMSE_lr_cf_valid = np.sqrt(mean_squared_error(y_cf_valid, lr_cf.predict(X_cf_valid)))

plt.text(50., 20., r'RMSE: %5.2f (validation)' %RMSE_lr_cf_valid, fontsize=15.)



plt.legend()



# --------------- RANDOM FORREST

# On Cleaned data

plt.figure(3)

xx,yy     = np.mgrid[y_c_valid.min():y_c_valid.max():nbins*1j, rf_c_prediction.min():rf_c_prediction.max():nbins*1j]

positions = np.vstack([xx.ravel(), yy.ravel()])

values    = np.vstack([y_c_valid, rf_c_prediction])

kernel    = stats.gaussian_kde(values)

zz        = np.reshape(kernel(positions).T, xx.shape)

plt.contour(xx, yy, zz, nlevels)

plt.scatter(y_c_valid, rf_c_prediction, s=1, label='RF_C')

plt.xlim(0, 80)

plt.ylim(0, 80)

plt.plot([0, 80], [0, 80], ls="--", c=".3")

plt.xlabel("Known Prices (validation data)")

plt.ylabel("Predicted Prices (validation data)")



# Make sure to write associated scores and test statistics

score_rf_c_train = rf_c.score(X_c_train, y_c_train)

plt.text(50., 40., r'Score: %5.2f (training)' %score_rf_c_train, fontsize=15.)

score_rf_c_valid = rf_c.score(X_c_valid, y_c_valid)

plt.text(50., 35., r'Score: %5.2f (validation)' %score_rf_c_valid, fontsize=15.)

RMSE_rf_c_train = np.sqrt(mean_squared_error(y_c_train, rf_c.predict(X_c_train)))

plt.text(50., 25., r'RMSE: %5.2f (training)' %RMSE_rf_c_train, fontsize=15.)

RMSE_rf_c_valid = np.sqrt(mean_squared_error(y_c_valid, rf_c.predict(X_c_valid)))

plt.text(50., 20., r'RMSE: %5.2f (validation)' %RMSE_rf_c_valid, fontsize=15.)



plt.legend()



# On Cleaned + featured data

plt.figure(4)

xx,yy     = np.mgrid[y_cf_valid.min():y_cf_valid.max():nbins*1j, rf_cf_prediction.min():rf_cf_prediction.max():nbins*1j]

positions = np.vstack([xx.ravel(), yy.ravel()])

values    = np.vstack([y_cf_valid, rf_cf_prediction])

kernel    = stats.gaussian_kde(values)

zz        = np.reshape(kernel(positions).T, xx.shape)

plt.contour(xx, yy, zz, nlevels)

plt.scatter(y_cf_valid, rf_cf_prediction, s=1, label='RF_CF')

plt.xlim(0, 80)

plt.ylim(0, 80)

plt.plot([0, 80], [0, 80], ls="--", c=".3")

plt.xlabel("Known Prices (validation data)")

plt.ylabel("Predicted Prices (validation data)")



# Make sure to write associated scores and test statistics

score_rf_cf_train = rf_cf.score(X_cf_train, y_cf_train)

plt.text(50., 40., r'Score: %5.2f (training)' %score_rf_cf_train, fontsize=15.)

score_rf_cf_valid = rf_cf.score(X_cf_valid, y_cf_valid)

plt.text(50., 35., r'Score: %5.2f (validation)' %score_rf_cf_valid, fontsize=15.)

RMSE_rf_cf_train = np.sqrt(mean_squared_error(y_cf_train, rf_cf.predict(X_cf_train)))

plt.text(50., 25., r'RMSE: %5.2f (training)' %RMSE_rf_cf_train, fontsize=15.)

RMSE_rf_cf_valid = np.sqrt(mean_squared_error(y_cf_valid, rf_cf.predict(X_cf_valid)))

plt.text(50., 20., r'RMSE: %5.2f (validation)' %RMSE_rf_cf_valid, fontsize=15.)



plt.legend()



# Show all plots

plt.show()
if tune_hyperparams:

    

    """

    This option allows us to identify the best model parameter set. This is done by

    defining a dictionary with the relevant hyper-parameters and the ranges to be

    explored. This dictionary, together with the model, is then passed into a grid

    search algorithm which performs the heavy lifting.

    """

    

    model = RandomForestRegressor()

    param_grid = {"n_estimators" : np.arange(50, 1100, 50),

                  "max_depth" : np.arange(1,51, 5),

                  "criterion" : ["mse"],

                  "min_samples_leaf" : [3],

                  "min_samples_split" : [3],

                  "bootstrap" : [True]

                  }



    random_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

    random_cv = random_cv.fit(X_cf_train, y_cf_train)



    print(random_cv.best_score_)

    print(random_cv.best_params_)

if show_parameter_distributions:

    

    """

    This option allows us to view individual parameter distributions. I originally used it to explore each

    parameter of the input before, and the effect of cleaning and adding features.  

    """

    

    dist_list = [xx for xx in df_c.columns if df_c[xx].dtype!='datetime64[ns]']

    

    for ii in dist_list:

        

        plt.figure()

        df[ii].plot(kind='hist', bins=200, range=(math.floor(df_cf[ii].min()),math.ceil(df_cf[ii].max())), alpha=0.2, density=True, label="df")

        df_c[ii].plot(kind='hist', bins=200, range=(math.floor(df_cf[ii].min()),math.ceil(df_cf[ii].max())), alpha=0.2, density=True, label="df_c")

        df_cf[ii].plot(kind='hist', bins=200, range=(math.floor(df_cf[ii].min()),math.ceil(df_cf[ii].max())), alpha=0.2, density=True, label="df_cf")

        plt.legend()

        plt.xlabel(ii)

    

    plt.show()
if show_feature_rankings:

    

    """

    This option allows us to look at the relative importance of each feature to

    the final taxi fare prediction. The values are based on the random forest

    output.

    """

    

    # On clean data

    plt.figure()

    features = X_c_train.columns

    importances = rf_c.feature_importances_

    indicies = np.argsort(importances)

    plt.barh(range(len(indicies)), importances[indicies], color='b', alpha=0.6, align='center')

    plt.title('Clean Data')

    plt.xlabel('Relative Importance')

    plt.yticks(range(len(indicies)), [features[i] for i in indicies])

    

    # On clean, feature-engineered data

    plt.figure()

    features = X_cf_train.columns

    importances = rf_cf.feature_importances_

    indicies = np.argsort(importances)

    plt.barh(range(len(indicies)), importances[indicies], color='g', alpha=0.6, align='center')

    plt.title('Clean + Feature Engineered Data')

    plt.xlabel('Relative Importance')

    plt.yticks(range(len(indicies)), [features[i] for i in indicies])

    

    plt.show()





    

    

    

    