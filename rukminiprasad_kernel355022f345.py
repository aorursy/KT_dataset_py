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
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv'
)
train

submission = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
test
submission
train.info()
print(train.describe())

print(test.describe())
data_Clear= train.drop(['Province_State','Id'],axis=1)
data_Clear
total_cases = data_Clear.groupby('Country_Region')['ConfirmedCases'].max().sort_values(ascending=False).to_frame()
total_cases.style.background_gradient(cmap='Blues')
train.plot.line( x ="Date",y="ConfirmedCases",title="Global Confirmed Cases")
latest.plot.bar( x ="Id",y="ConfirmedCases",title="Global Confirmed Cases")
group = train.groupby(["Country_Region", "Date"]).sum()
latest = group.groupby(["Country_Region"]).last()
latest.head()
most_cases = latest.sort_values(by="ConfirmedCases", ascending=False).head(10)
most_cases.head()
import pandas as pd
import numpy as np
sub_df = pd.DataFrame({'ForecastId':[],
                      'ConfirmedCases': [],
                      'Fatalities': []
                      },dtype=np.int64)
print(sub_df.describe())
print(sub_df.isnull().sum())
for one_area in area_list:
    X_train = train_df.loc[train_df['Area'] == one_area]
    X_test = test_df.loc[test_df['Area'] == one_area]
    xs = range(0, X_train.shape[0])
    y_train_case = list(X_train['ConfirmedCases'])
    y_train_fat = list(X_train['Fatalities'])
    forecastxs = range(X_train.shape[0], X_train.shape[0] + X_test.shape[0])
if (len(X_train['ConfirmedCases'].unique()) > 4):
        
        # first guess as to the values needed in the logistic function, for curve_fit
        case_p0 = [1000000, 25, -.1] 
        # fit a logistic curve for case count with 10k iterations and initial values stored in p0
        case_opt, case_cov = curve_fit(logistic, xs, y_train_case, maxfev=500000, p0=case_p0)
    
        y_fitted_train_case = np.round(logistic(xs, case_opt[0], case_opt[1], case_opt[2]), 0)

        # forecast the values for case count from the curve we just fit
        y_pred_case = np.round(logistic(forecastxs, case_opt[0], case_opt[1], case_opt[2]), 0)    
    
        # calculate the value of 1 std dev for each of those measures
        case_sd = np.sqrt(np.diag(case_cov))

        
        low_y_pred_case = np.round(logistic(forecastxs, case_opt[0]-case_sd[0], case_opt[1]-case_sd[1], case_opt[2]-case_sd[2]))
        high_y_pred_case = np.round(logistic(forecastxs, case_opt[0]+case_sd[0], case_opt[1]+case_sd[1], case_opt[2]+case_sd[2]))
             
else:
        m, b, r, p, std_err = linregress(xs, y_train_case)
        y_fitted_train_case = np.maximum(np.zeros(len(xs)), np.round((m * xs) + b, 0))
        y_pred_case = np.round((m * forecastxs) + b, 0)
        fatality_ratio = np.mean(list(X_train.loc[np.isnan(X_train['FatalityRatio']) == False, 'FatalityRatio']))
if np.isnan(fatality_ratio) == True:
        
        # use the global average for this country
        fatality_ratio = X_train['FatalityRatio'].mean()
        
y_fat_ratio_train = np.round(X_train['ConfirmedCases'] * fatality_ratio, 0)
y_fat_ratio_forecast = np.round(fatality_ratio * y_pred_case, 0)
ids = test_df.loc[test_df['Area'] == one_area, 'ForecastId']
sub_df = pd.concat([sub_df, pd.DataFrame({'ForecastId' : ids,
                                            'ConfirmedCases' : y_pred_case,
                                            'Fatalities' : y_fat_ratio_forecast
                                            },dtype=np.int64)])
    
train_case_rmsle = rmsle(y_train_case, y_fitted_train_case)
print("{0} rmsle cases: {1:.3f}".format(one_area, train_case_rmsle))
    
        
sub_df['Fatalities'] = sub_df['Fatalities'].fillna(value=0)
sub_df['Fatalities'] = sub_df['Fatalities'].astype('int64')
print(sub_df.describe())
print(sub_df.isnull().sum())
print(sub_df.info())
print(sub_df.shape)

sub_df['Fatalities'] = sub_df['Fatalities'].fillna(value=0)
sub_df['Fatalities'] = sub_df['Fatalities'].astype('int64')
print(sub_df.describe())
print(sub_df.isnull().sum())
print(sub_df.info())
print(sub_df.shape)
sub_df.to_csv('submission.csv', header=True, index=False)
print("Complete.")