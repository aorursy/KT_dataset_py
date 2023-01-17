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

import numpy as np



import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



def preprocess(total, longlat = True, lags = True, dummies = False):



    df = total.copy()



    df['Province_State'].fillna('-', inplace = True)

    df['Date'] = pd.to_datetime(df['Date'])

    df['Weekday'] = df['Date'].apply(lambda x: x.weekday())

    df['t'] = (df['Date'] - df['Date'].min()) / pd.to_timedelta(1, unit='D')



    if longlat:

        longlat = pd.read_csv('/kaggle/input/longlat/train_longlat.csv')

        longlat['Province/State'].fillna('-', inplace = True)

        longlat = longlat.groupby(['Province/State', 'Country/Region']).first()[['Lat', 'Long']]

        longlat = longlat.reset_index()

        df = df.merge(longlat, how = 'left', left_on = ['Province_State', 'Country_Region'], right_on = ['Province/State', 'Country/Region'])



    groups = df.groupby(['Country_Region', 'Province_State'])



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



    # Add DailyCases

    df['DailyCases'] = pd.concat([group['ConfirmedCases'].diff(1).copy() for (name, group) in groups], axis = 0)



    # Add lag variables

    if lags is True:

        df['DailyCases_L1'] = pd.concat([group['DailyCases'].shift(1).copy() for (name, group) in groups], axis = 0)

        df['ConfirmedCases_L1'] = pd.concat([group['ConfirmedCases'].shift(1).copy() for (name, group) in groups], axis = 0)

        df['Fatalities_L1'] = pd.concat([group['Fatalities'].shift(1).copy() for (name, group) in groups], axis = 0)



    if dummies:

        dummies = pd.get_dummies(df['Country_Region'])

        df = df.join(dummies)



    return df



def train_test_split(total, daily_cases = False, lags = False):



    #date_train_last = '2020-03-19'

    #date_test_first = '2020-03-20'



    total_copy = total.copy()

    train = total_copy.loc[total_copy['ForecastId'].isnull()]



    # remove unknown variables from test

    test = total.loc[total['ForecastId'].notnull()].copy()

    if not daily_cases:

        test['DailyCases'] = np.nan

    if lags:

        test['ConfirmedCases_L1'] = np.nan

        test['DailyCases_L1'] = np.nan

        test['Fatalities_L1'] = np.nan



    print('Train timeframe : {} - {}'.format(train['Date'].min(), train['Date'].max()))

    print('Train shape : {}'.format(train.shape))



    print('Test timeframe  : {} - {}'.format(test['Date'].min(), test['Date'].max()))

    print('Test shape : {}'.format(test.shape))



    return train, test



def print_feature_importances(clf, index):

    return pd.Series(clf.feature_importances_, index = index).sort_values(ascending = False)



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



def plot_region_pred(group, test_, dependant = 'ConfirmedCases'):



    test_gr = test_.groupby(['Country_Region', 'Province_State'])

    gr = test_gr.get_group(group)

    gr.set_index('Date', inplace = True)



    f, ax = plt.subplots(figsize=(7, 7))

    gr[[dependant, dependant+'_pred']].plot()

import pandas as pd

import numpy as np



train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')



print('Train timeframe : {} - {}'.format(train['Date'].min(), train['Date'].max()))

print('Train shape : {}'.format(train.shape))



print('Test timeframe  : {} - {}'.format(test['Date'].min(), test['Date'].max()))

print('Test shape : {}'.format(test.shape))



total = train.merge(test, how = 'outer', on = ['Province_State', 'Country_Region', 'Date'])

total = preprocess(total, longlat = True, lags = False)

train, test = train_test_split(total, daily_cases = True, lags = False)

total.info()
def preprocess_train_RFC(train, regressors, daily_cases = False, lags = False):

    

    if daily_cases:

        train = train.loc[train['DailyCases'].notnull()]

        

    X_train = train[regressors].copy()

    c_train = train[dependant].copy()

    

    if lags:

        X_train['ConfirmedCases_L1'] = X_train['ConfirmedCases_L1'].fillna(0)

        X_train['Fatalities_L1'] = X_train['Fatalities_L1'].fillna(0)

    

    X_train['Lat'] = X_train['Lat'].fillna(0)

    X_train['Long'] = X_train['Long'].fillna(0)

    

    return X_train, c_train



def preprocess_test_RFC(test, regressors, dependant):

    

    X_test = test[regressors].copy()

    c_test = test[dependant].copy()

        

    X_test['Lat'] = X_test['Lat'].fillna(0)

    X_test['Long'] = X_test['Long'].fillna(0)

    

    return X_test, c_test



dependant = 'ConfirmedCases'

regressors = ['Lat', 'Long', 'Weekday', 'TTF', 't']

#regressors = ['Weekday', 'TTF', 't']



X_train, c_train = preprocess_train_RFC(train, regressors, daily_cases=False)

X_test, c_test = preprocess_test_RFC(test, regressors, dependant)

X_train.info()
# Train the RandomForestClassifier



from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(max_depth=200, random_state=0)

rfc.fit(X_train, c_train)

c_train_pred = rfc.predict(X_train)

print_feature_importances(rfc, X_train.columns)
# Predict ConfirmedCases



c_hat_test = rfc.predict(X_test)

c_hat_test = pd.Series(c_hat_test, index = X_test.index, name = 'ConfirmedCases_pred')



from sklearn.metrics import mean_squared_log_error



test_ = test.join(c_hat_test).loc[test['ConfirmedCases'].notnull()]

RMSLE_test = np.sqrt(mean_squared_log_error(test_['ConfirmedCases'], test_["ConfirmedCases_pred"]))

RMSLE_train = np.sqrt(mean_squared_log_error(train['ConfirmedCases'], c_train))                                              



# Test / train errors

print('RMSLE (train) : {}'.format(RMSLE_train))

print('RMSLE (test) : {}'.format(RMSLE_test))



# Compute test error wrt horizon

for t in test_['t'].unique():

    test_t = test_.loc[test_['t'] == t]

    RMSLE_test = np.sqrt(mean_squared_log_error(test_t['ConfirmedCases'], test_t["ConfirmedCases_pred"]))

    print('RMSLE (horizon t={}) : {}'.format(int(t), RMSLE_test))

    

# Compute test error wrt Region



RMSE_region = []

test_gr = test_.groupby(['Country_Region', 'Province_State'])

for key in test_gr.groups.keys():

    test_region = test_gr.get_group(key)

    RMSE = np.sqrt(mean_squared_log_error(test_region['ConfirmedCases'], test_region["ConfirmedCases_pred"]))

    RMSE_region.append(RMSE)

RMSE_region = pd.Series(RMSE_region, index = test_gr.groups.keys())

RMSE_region.sort_values(ascending = True).tail(30).plot.bar()
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



def plot_region_pred(group, test_, dependant = 'ConfirmedCases'):



    test_gr = test_.groupby(['Country_Region', 'Province_State'])

    

    gr = test_gr.get_group(group)

    gr.set_index('Date', inplace = True)



    #fig, axs = plt.subplots(2,2, figsize=(15, 15), facecolor='w', edgecolor='k')



    gr[[dependant, dependant+'_pred']].plot.bar(title = group[0] + '/' + group[1])

    #gr[dependant+'_pred'].hist(ax = axs[1])

    return gr



region = ('France', '-')

region = ('China', 'Hong Kong')



gr = plot_region_pred(region, test_, 'ConfirmedCases')
obs_vs_pred_plot(c_train, c_train_pred, c_test, c_hat_test)
dependant = 'Fatalities'

regressors = ['Long', 'Lat', 'Weekday', 'TTF', 't']



X_train, f_train = preprocess_train_RFC(train, regressors, daily_cases=False)

X_test, f_test = preprocess_test_RFC(test, regressors, dependant)

rfc_fatalities = RandomForestClassifier(max_depth=200, random_state=0)

rfc_fatalities.fit(X_train, f_train)

f_train_pred = rfc_fatalities.predict(X_train)

print_feature_importances(rfc, X_train.columns)



f_hat_test = rfc_fatalities.predict(X_test)

f_hat_test = pd.Series(f_hat_test, index = X_test.index, name = 'Fatalities_pred')
test_ = test.join(f_hat_test).loc[test['Fatalities'].notnull()]

RMSLE_test = np.sqrt(mean_squared_log_error(test_['Fatalities'], test_["Fatalities_pred"]))



print('RMSLE (test) : {}'.format(RMSLE_test))



# Compute test error wrt horizon

for t in test_['t'].unique():

    test_t = test_.loc[test_['t'] == t]

    RMSLE_test = np.sqrt(mean_squared_log_error(test_t['Fatalities'], test_t["Fatalities_pred"]))

    print('RMSLE (horizon t={}) : {}'.format(int(t), RMSLE_test))

    

# Compute test error wrt Region

RMSE_region = []

test_gr = test_.groupby(['Country_Region', 'Province_State'])

for key in test_gr.groups.keys():

    test_region = test_gr.get_group(key)

    RMSE = np.sqrt(mean_squared_log_error(test_region['Fatalities'], test_region["Fatalities_pred"]))

    RMSE_region.append(RMSE)

RMSE_region = pd.Series(RMSE_region, index = test_gr.groups.keys())

RMSE_region.sort_values(ascending = True).tail(30).plot.bar()
gr = plot_region_pred(region, test_, 'Fatalities')
sub = pd.concat([test['ForecastId'], c_hat_test, f_hat_test], axis = 1)

sub.columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']

sub['ForecastId'] = sub['ForecastId'].apply(int)

sub.set_index('ForecastId', inplace = True)

sub.to_csv('submission.csv')

sub