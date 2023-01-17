# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

train['Province/State'].fillna('-', inplace = True)

train['Date'] = pd.to_datetime(train['Date'])



test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

test['Province/State'].fillna('-', inplace = True)

test['Date'] = pd.to_datetime(test['Date'])



sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

sub['ForecastId'] = test['ForecastId'].values

sub = sub.set_index('ForecastId', drop=True)



# REMOVE OVERLAP TRAIN / TEST



to_test = train[train["Date"] >= min(test['Date'])]

test = test.merge(to_test, how='left', on=['Date', 'Country/Region', 'Province/State', 'Lat', 'Long'])

test.describe()

train = train[train["Date"] < min(test['Date'])]



print('Train timeframe : {} - {}'.format(train['Date'].min(), train['Date'].max()))

print('Train shape : {}'.format(train.shape))



print('Test timeframe  : {} - {}'.format(test['Date'].min(), test['Date'].max()))

print('Test shape : {}'.format(test.shape))

train.head()
import numpy as np

import statsmodels.api as sm

from sklearn.preprocessing import OneHotEncoder

import datetime



def preprocess(df, dependant, add_constant = True, dummies = True):

    

    df = df.copy()

    

    df['Weekday'] = df['Date'].apply(lambda x: x.weekday())



    groups = df.groupby(['Country/Region'])

    #groups = df.groupby(['Country/Region', 'Province/State'])

    

    # COMPUTE DAILY CASES, ETC

    

    groups_ = []

    for name,group in groups:



        group = group.copy()

        group['DailyCases'] = group['ConfirmedCases'].diff(1)    

        group['DailyCases_L1'] = group['DailyCases'].shift(1)

        group['ConfirmedCases_L1'] = group['ConfirmedCases'].shift(1)

        group['Fatalities_L1'] = group['Fatalities'].shift(1)

        groups_.append(group)



    df = pd.concat(groups_, axis = 0)

    

    # COMPUTE TIME TO FIRST CASE

    ttf_vec = []

    for name,group in groups:



        first_case_index = group['ConfirmedCases'].loc[group['ConfirmedCases'] > 0].first_valid_index()

        if first_case_index is not None:

            first_case_date = group.loc[first_case_index]['Date']

            ttf = (group['Date'] - first_case_date) / pd.to_timedelta(1, unit='D')

            ttf.loc[ttf<0] = 0

        else:

            ttf = pd.Series([0]*len(group))

        ttf.index = group.index

        ttf_vec.append(ttf)

    

    df['TTF'] = pd.concat(ttf_vec)

    

    if dummies:

        dummies = pd.get_dummies(df['Country/Region'])

        df = df.join(dummies)

    

    df.drop(['Country/Region', 'Province/State'], axis = 1, inplace = True)

    

    df.dropna(inplace = True)

    

    y = df[dependant]

    df.drop(['Date', 'Lat', 'Long', 'Id', 'ConfirmedCases', 'DailyCases', 'Fatalities'], axis = 1, inplace = True)



    X = df

    if add_constant:

        X = sm.add_constant(df, has_constant = 'add')

 

    return X, y, df



def obs_vs_pred_plot(y_train, y_pred_train, y_test, y_pred_test, title = None):



    plot_train = pd.concat([y_train, pd.Series(y_pred_train, index = y_train.index)], axis = 1)

    plot_train.columns = ["Observations", "Predictions"]



    plot_test = pd.concat([y_test, pd.Series(y_pred_test, index = y_test.index)], axis = 1)

    plot_test.columns = ["Observations", "Predictions"]



    f, ax = plt.subplots(figsize=(7, 7))

    plt.title(title)

    

    sc = sns.scatterplot(x="Observations", y="Predictions", data = plot_train, label = "Train")

    sc.axes.set_ylim(0,max(plot_train.max().max(), plot_test.max().max()))

    sc.axes.set_xlim(0,max(plot_train.max().max(), plot_test.max().max()))



    sns.scatterplot(x="Observations", y="Predictions", data = plot_test, label = "Test")

    return None
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error



X_train_cc, y_train_cc, df_cc = preprocess(train, dependant = 'ConfirmedCases', add_constant=False, dummies=False)

X_test_cc, y_test_cc, df_test_cc = preprocess(test, dependant='ConfirmedCases', add_constant=False, dummies=False)

X_test_cc.drop('ForecastId', axis = 1, inplace = True)



rf = RandomForestRegressor(max_depth=20, random_state=0)

rf.fit(X_train_cc, y_train_cc)

print("In-sample score (R2): {}".format(rf.score(X_train_cc, y_train_cc)))



y_pred_train_cc = rf.predict(X_train_cc)

y_pred_test_cc = rf.predict(X_test_cc)



score_cc_train = np.sqrt(mean_squared_log_error(y_pred_train_cc, y_train_cc))

score_cc_test = np.sqrt(mean_squared_log_error(y_pred_test_cc, y_test_cc))



print('Mean squared log error (train) : {}'.format(score_cc_train))

print('Mean squared log error (test) : {}'.format(score_cc_test))



obs_vs_pred_plot(y_train_cc, y_pred_train_cc, y_test_cc, y_pred_test_cc, title = "RandomForest pour ConfirmedCases")



print(pd.Series(rf.feature_importances_, index = X_train_cc.columns).sort_values(ascending = False).head(15))
X_train_fat, y_train_fat, df_fat = preprocess(train, dependant = 'Fatalities', add_constant=False, dummies=True)

X_test_fat, y_test_fat, df_fat = preprocess(test, dependant='Fatalities', add_constant=False, dummies=True)

X_test_fat.drop('ForecastId', axis = 1, inplace = True)



rf = RandomForestRegressor(max_depth=20, random_state=0)

rf.fit(X_train_fat, y_train_fat)

print("In-sample score (R2): {}".format(rf.score(X_train_fat, y_train_fat)))



print(pd.Series(rf.feature_importances_, index = X_train_fat.columns).sort_values(ascending = False).head(15))



y_pred_train_fat = rf.predict(X_train_fat)

y_pred_test_fat = rf.predict(X_test_fat)



def RMSLE(y,y_):

    return np.sqrt(mean_squared_log_error(y, y_))



score_c_train = RMSLE(y_pred_train_fat, y_train_fat)

score_c_test = RMSLE(y_pred_test_fat, y_test_fat)



print('Mean squared log error (train) : {}'.format(score_c_train))

print('Mean squared log error (test) : {}'.format(score_c_test))



obs_vs_pred_plot(y_train_fat, y_pred_train_fat, y_test_fat, y_pred_test_fat, title = "RandomForest pour Fatalities")
sub_ = pd.concat([pd.Series(y_pred_test_cc), pd.Series(y_pred_test_fat)], axis = 1)

sub_.columns = ['ConfirmedCases', 'Fatalities']

sub_['ForecastId'] = y_test_cc.index

sub_.set_index('ForecastId', inplace = True)



sub.loc[sub_.index] = sub_

sub.to_csv('submission.csv')

sub.head()