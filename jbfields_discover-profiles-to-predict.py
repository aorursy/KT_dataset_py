# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.tseries.frequencies import to_offset #Set the frequency in the index

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Used to set the index on the dataframes
def return_set_index(X):
    X_ = X.copy()
    X_['datetime'] = pd.to_datetime(X_['datetime'], infer_datetime_format=True)
    
    X_ = X_.set_index('datetime')
    X_ = X_.sort_index()
    X_.freq = pd.tseries.frequencies.to_offset('H')
    return X_
#This function will be helpful to establish the profiles. It takes two different groups X_workingday, X_holiday and groups by the hour. For each hour, for each group
#The mean, std and the ratio is compared.
def establish_profile(X_workingday, X_holiday, X_total):
    info_mean_workingday = X_workingday['count'].groupby(X_workingday['count'].index.hour).mean()
    info_std_workingday = X_workingday['count'].groupby(X_workingday['count'].index.hour).std()

    info_mean_holiday = X_holiday['count'].groupby(X_holiday['count'].index.hour).mean()
    info_std_holiday = X_holiday['count'].groupby(X_holiday['count'].index.hour).std()

    info_mean_total = X_total['count'].groupby(X_total['count'].index.hour).mean()
    info_std_total = X_total['count'].groupby(X_total['count'].index.hour).std()


    statistic_info = pd.DataFrame({'mean_workingday' : info_mean_workingday, 'std_workingday': info_std_workingday,
                               'mean_holiday' : info_mean_holiday, 'std_holiday': info_std_holiday,
                               'mean_total' : info_mean_total, 'std_total': info_std_total
                              })
    statistic_info['mean_over_std_workingday'] = statistic_info['mean_workingday']/statistic_info['std_workingday']
    statistic_info['mean_over_std_holiday'] = statistic_info['mean_holiday']/statistic_info['std_holiday']
    statistic_info['mean_over_std_total'] = statistic_info['mean_total']/statistic_info['std_total']
    return statistic_info
#Having the statistical information between 3 different groups, this function counts, for each group, the number of hours thats saw an improvement in the ratio mean/std
def how_good_profile(statistic_info, 
                     feature_weekday ='mean_over_std_workingday', 
                     feature_weekend ='mean_over_std_holiday', 
                     feature_total   ='mean_over_std_total'):
    n_hours = statistic_info.shape[0]
    w_day = (statistic_info[feature_weekday] > statistic_info[feature_total]).sum()/n_hours
    w_hol = (statistic_info[feature_weekend] > statistic_info[feature_total]).sum()/n_hours
    return (w_day, w_hol)
## Auxiliary function that creates a column corresponding to the hour
def hour_column_id(X):
    X_ = X.copy()
    X_['dt'] = X_.index.astype(str)
    X_['new_dt'] = X_['dt'].str.split(' ', expand=True)[1]
    X_['hour'] = X_['new_dt'].map(lambda x: str(x)[:2])
    X_['hour'] = X_['hour'].astype(int)
    return X_.drop(columns=['dt', 'new_dt'])

## Function used to make the forecast. It must take the dates for which the forecast is to be considered and the right statistical info for that time
def make_forecast(X_test, statistical_info, feature):
    X_t = hour_column_id(X_test)
    X_r = X_t.join(statistical_info, on='hour')
    X_r['count'] = X_r[feature].round().astype(int)
    return pd.DataFrame(X_r['count'])
#Import the files and set the index
X_train = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')
X_train = return_set_index(X_train)
X_test = return_set_index(X_test)

#We will devide the train dataset in seasons and working days vs non working days. The idea is to group the results by hour and observe if the ratio 
#mean standard deviation for each group is better or not for the overall group.
X_workingday_1 = X_train[(X_train.workingday == 1) & (X_train.season == 1)]
X_holiday_1    = X_train[(X_train.workingday == 0) & (X_train.season == 1)]
X_total_1      = X_train[(X_train.season == 1)]

X_workingday_2 = X_train[(X_train.workingday == 1) & (X_train.season == 2)]
X_holiday_2    = X_train[(X_train.workingday == 0) & (X_train.season == 2)]
X_total_2      = X_train[(X_train.season == 2)]

X_workingday_3 = X_train[(X_train.workingday == 1) & (X_train.season == 3)]
X_holiday_3    = X_train[(X_train.workingday == 0) & (X_train.season == 3)]
X_total_3      = X_train[(X_train.season == 3)]


X_workingday_4 = X_train[(X_train.workingday == 1) & (X_train.season == 4)]
X_holiday_4    = X_train[(X_train.workingday == 0) & (X_train.season == 4)]
X_total_4      = X_train[(X_train.season == 4)]
statistic_info_1 = establish_profile(X_workingday_1, X_holiday_1, X_total_1)
statistic_info_2 = establish_profile(X_workingday_2, X_holiday_2, X_total_2)
statistic_info_3 = establish_profile(X_workingday_3, X_holiday_3, X_total_3)
statistic_info_4 = establish_profile(X_workingday_4, X_holiday_4, X_total_4)
statistic_info_t = establish_profile(X_train[(X_train.workingday == 1)], X_train[(X_train.workingday == 0)], X_train)

#We can observe that this separation increases the ratio mean/std at least for working days. The same cannot always be said about holidays. Thus it is important to
#write a function that counts the amount of hours for which this separation increased the ratio mean/std. For working days vs non working days and for season
statistic_info_1
#This information tells us that, for working days, the ratio mean/std for counts for each hour improved when compared to the same ratio for all the days confounded. However, for non working days,
# we can see that this separation worked well for working days but not so well for non working days.
# The percentage of hours for which working days have a better ratio mean counts/std counts than all the days confounded is:
#Season 1 (100%)
#Season 2 (100%)
#Season 3 (95.83%)
#Season 4 (87.5%)
#All seasons (100%)

# The percentage of hours for which NON working days have a better ratio mean counts/std counts than all the days confounded is:
#Season 1 (45.83%)
#Season 2 (50%)
#Season 3 (79.16%)
#Season 4 (70.83%)
#All seasons (50%)
print(how_good_profile(statistic_info_1))
print(how_good_profile(statistic_info_2))
print(how_good_profile(statistic_info_3))
print(how_good_profile(statistic_info_4))
print(how_good_profile(statistic_info_t))
# It is undeniable that working days have a distinct behavior on their own. So we assume that the separation non working days vs working days is a valid one. However,
# Looking at the non working days the conclusion is less obvious. We will assume that for seasons 3 and 4 we have a distinct pattern for non working days and leave a 
# possible refinement to a later version. Lets see if we can find a separation for weekends and public holidays for seasons 1 and 2.

# In what follows we will repeat the steps above but now, instead of X_train we will have X_holiday_1, instead of X_workingday_1 we will have X_1_public_holidays 
# and X_1_weekends instead X_holiday_1. And the same for season 2. We try to figure out if this separation will increase the ration mean counts /std counts at least
# for one of these groups
X_1_weekends = X_train.loc[(X_train.season == 1) & 
                            ((X_train.index.weekday_name == 'Saturday') | 
                             (X_train.index.weekday_name == 'Sunday'))]
    
X_1_public_holidays = X_train.loc[(X_train.season == 1) & 
                                  (X_train.workingday==0) & 
                                  (X_train.index.weekday_name != 'Saturday') & 
                                  (X_train.index.weekday_name != 'Sunday')]

X_2_weekends = X_train.loc[(X_train.season == 2) & 
                            ((X_train.index.weekday_name == 'Saturday') | 
                             (X_train.index.weekday_name == 'Sunday'))]
    
X_2_public_holidays = X_train.loc[(X_train.season == 2) & 
                                  (X_train.workingday==0) & 
                                  (X_train.index.weekday_name != 'Saturday') & 
                                  (X_train.index.weekday_name != 'Sunday')]
## 2nd profile level: Non working days for seasons 1 and 2
# X_workingday_1 : X_1_public_holidays
# X_holiday_1 : X_1_weekends
# X_total_1 : X_holiday_1
# And the same for season 2
statistic_info_no_workingdays_1 = establish_profile(X_1_public_holidays, X_1_weekends, X_holiday_1)
statistic_info_no_workingdays_2 = establish_profile(X_2_public_holidays, X_2_weekends, X_holiday_2)

print(how_good_profile(statistic_info_no_workingdays_1))
print(how_good_profile(statistic_info_no_workingdays_2))
# Conclusion: We can observe that, for season 1, by separating the public holidays from the weekends we observe a better ratio for the public holidays, while for
# weekends this ratio is higher for only 50% of the hours, while for season 2, the conclusion is the opposite.
# It is now time to make the forecast for the different profiles considered
# Working days
X_workingday_test_1 = X_test[(X_test.workingday == 1) & (X_test.season == 1)]
X_workingday_test_2 = X_test[(X_test.workingday == 1) & (X_test.season == 2)]
X_workingday_test_3 = X_test[(X_test.workingday == 1) & (X_test.season == 3)]
X_workingday_test_4 = X_test[(X_test.workingday == 1) & (X_test.season == 4)]

X_n_workingday_test_3 = X_test[(X_test.workingday == 0) & (X_test.season == 3)]
X_n_workingday_test_4 = X_test[(X_test.workingday == 0) & (X_test.season == 4)]

#First partition
X_wd_test_1 = make_forecast(X_workingday_test_1, statistic_info_1, 'mean_workingday')
X_wd_test_2 = make_forecast(X_workingday_test_2, statistic_info_2, 'mean_workingday')
X_wd_test_3 = make_forecast(X_workingday_test_3, statistic_info_3, 'mean_workingday')
X_wd_test_4 = make_forecast(X_workingday_test_4, statistic_info_4, 'mean_workingday')

X_n_wd_test_3 = make_forecast(X_n_workingday_test_3, statistic_info_3, 'mean_holiday')
X_n_wd_test_4 = make_forecast(X_n_workingday_test_4, statistic_info_4, 'mean_holiday')

#Second partition
X_1_weekends_test = X_test.loc[(X_test.season == 1) & 
                            ((X_test.index.weekday_name == 'Saturday') | 
                             (X_test.index.weekday_name == 'Sunday'))]
    
X_1_public_holidays_test = X_test.loc[(X_test.season == 1) & 
                                  (X_test.workingday==0) & 
                                  (X_test.index.weekday_name != 'Saturday') & 
                                  (X_test.index.weekday_name != 'Sunday')]

X_2_weekends_test = X_test.loc[(X_test.season == 2) & 
                            ((X_test.index.weekday_name == 'Saturday') | 
                             (X_test.index.weekday_name == 'Sunday'))]
    
X_2_public_holidays_test = X_test.loc[(X_test.season == 2) & 
                                  (X_test.workingday==0) & 
                                  (X_test.index.weekday_name != 'Saturday') & 
                                  (X_test.index.weekday_name != 'Sunday')]

X_1_weekend_test_forecast = make_forecast(X_1_weekends_test, statistic_info_no_workingdays_1, 'mean_holiday')
X_2_weekend_test_forecast = make_forecast(X_2_weekends_test, statistic_info_no_workingdays_2, 'mean_holiday')
X_1_public_holidays_test_forecast = make_forecast(X_1_public_holidays_test, statistic_info_no_workingdays_1, 'mean_holiday')
X_2_public_holidays_test_forecast = make_forecast(X_2_public_holidays_test, statistic_info_no_workingdays_2, 'mean_holiday')

# Final prediction
X_final = pd.concat([X_wd_test_1, 
                     X_wd_test_2, 
                     X_wd_test_3, 
                     X_wd_test_4, 
                     X_n_wd_test_3, 
                     X_n_wd_test_4, 
                     X_1_weekend_test_forecast, 
                     X_2_weekend_test_forecast, 
                     X_1_public_holidays_test_forecast, 
                     X_2_public_holidays_test_forecast])
X_final = X_final.sort_index()
X_final.reset_index(level=0, inplace=True)
X_final.to_csv('result.csv', index=False)
