# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
from scipy.optimize import curve_fit

# Load data
train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')

# Convert date columns to datetime
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

train['Country_Region'] = train['Country_Region'].apply(lambda x: 'Georgia (Cntry)' if x=='Georgia' else x)
test['Country_Region'] = test['Country_Region'].apply(lambda x: 'Georgia (Cntry)' if x=='Georgia' else x)

train['Province_State'] = train['Province_State'].fillna(train['Country_Region'])
test['Province_State'] = test['Province_State'].fillna(test['Country_Region'])

# Fix outliers
# Saint Barthelemy
idx_0 = train[(train.Province_State=='Saint Barthelemy') & (train.Date=='2020-03-04')].index[0]
idx_1 = train[(train.Province_State=='Saint Barthelemy') & (train.Date=='2020-03-08')].index[0]
for i in range(idx_0, idx_1 + 1):
    train.loc[i,'ConfirmedCases'] = 0

# Guyana
idx_0 = train[(train.Province_State=='Guyana') & (train.Date=='2020-03-17')].index[0]
idx_1 = train[(train.Province_State=='Guyana') & (train.Date=='2020-03-23')].index[0]
for i in range(idx_0, idx_1 + 1):
    train.loc[i,'ConfirmedCases'] = 4

# Iceland
idx_0 = train[(train.Province_State=='Iceland') & (train.Date=='2020-03-15')].index[0]
idx_1 = train[(train.Province_State=='Iceland') & (train.Date=='2020-03-20')].index[0]
train.loc[idx_0, 'Fatalities'] = 0
train.loc[idx_1, 'Fatalities'] = 1


def sigmoid(x, M, a, B):
    return M / (1 + np.exp(-B * (x - a)))

def fit_sigmoid(train, target):
    df = train[train[target] != 0]
    df = df[['Date', target]]
    st_date = df.Date.min()

    # Set boundaries
    M_min = df[target].max()
    M_max = M_min * 32     

    # Convert dates to day numbers
    df['Date'] = (df['Date'] - df['Date'].min()).dt.days

    # Fit to a sigmoid function
    popt, pcov = curve_fit(sigmoid, df['Date'], df[target],
                           bounds = ([M_min, 10, 0.0], [M_max, 75, 0.5]))
    M, a, B = popt

    region_param = pd.DataFrame(data={'M': M, 'a': a, 'B': B, 'st_date': st_date},
                                index=[REGION])
    return region_param


# Find sigmoid parameters of Confirmed Cases
sigmoid_params = pd.DataFrame([])
for REGION in train.Province_State.unique():
    region_param = fit_sigmoid(train[train.Province_State==REGION], 'ConfirmedCases')
    sigmoid_params = pd.concat([sigmoid_params, region_param], axis=0)

# Make final predictions for cases
cases = pd.DataFrame([])
for REGION in test.Province_State.unique():
    df = test[test.Province_State==REGION].copy()
    M, a, B, st_date = sigmoid_params.loc[REGION].values

    df['Day'] = (df['Date'] - st_date).dt.days
    df['ConfirmedCases'] = df['Day'].apply(lambda x: sigmoid(x, M, a, B))

    cases = pd.concat([cases, df], axis=0)

# Find sigmoid parameters of Fatalities
sigmoid_params = pd.DataFrame([])
for REGION in train.Province_State.unique():
    if train[train.Province_State==REGION]['Fatalities'].max() == 0:
        params = {'M': 0, 'a': 0, 'B': 0, 'st_date': train.Date.max()}
        region_param = pd.DataFrame(params, index=[REGION])
    else:
        region_param = fit_sigmoid(train[train.Province_State==REGION], 'Fatalities')
    sigmoid_params = pd.concat([sigmoid_params, region_param], axis=0)    

# Make final predictions for cases
fatalities = pd.DataFrame([])
for REGION in test.Province_State.unique():
    df = test[test.Province_State==REGION].copy()
    M, a, B, st_date = sigmoid_params.loc[REGION].values

    df['Day'] = (df['Date'] - st_date).dt.days
    df['Fatalities'] = df['Day'].apply(lambda x: sigmoid(x, M, a, B))

    fatalities = pd.concat([fatalities, df], axis=0)

# Prepare final submission
submission = pd.DataFrame({'ForecastId': test['ForecastId'],
                           'ConfirmedCases': cases['ConfirmedCases'],
                           'Fatalities': fatalities['Fatalities']})

submission.to_csv('submission.csv', index=False)

