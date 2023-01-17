from scipy.stats import ttest_ind as ttest

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
def _filter_priority(calls,prty):

    if prty == 'All':

        return calls

    return calls[calls['priority']==prty]
# This function counts the number of calls

# @param pandas Series of calls

# @param desired year

# @return pandas Series of call count per month

def count_it(calls, yr, prty, weekly=True):

    call_list = _filter_priority(calls,prty)

    date_series = pd.to_datetime(call_list['callDateTime'])



    if weekly:

        date_series = date_series[date_series.dt.year == yr]

        return date_series.dt.week.value_counts()



    if yr == 2017: mths = 7

    else: mths = 12



    counts = pd.Series(np.zeros(mths), index=range(1,mths+1))



    for mth in range(1,mths+1):

        counts.set_value(label = mth, value = date_series[(date_series.dt.year == yr) &

                                           (date_series.dt.month == mth)].count())



    return counts
# This function calculates the median number of calls per week

# @param pandas Series of calls

# @param desired priority or all calls

# @return pandas Series of median number of calls per week

def weekly_medians(calls,prty):

    call_list = _filter_priority(calls,prty)



    return pd.Series([count_it(call_list, 2015, prty).median(),

                      count_it(call_list, 2016, prty).median(),

                      count_it(call_list, 2017, prty).median()],

                     index=[2015,2016,2017],

                     name=prty)
# This function runs ttest on weekly number of calls

# @param pandas Series of calls

# @param desired year

# @return pandas Series of call count per month

def run_ttest(calls, prty):

    call_list = _filter_priority(calls,prty)



    return pd.Series([ttest(count_it(call_list,2015, prty),count_it(call_list,2016, prty))[1],

                      ttest(count_it(call_list, 2016, prty), count_it(call_list, 2017, prty))[1]],

                     index=[2016,2017],

                     name=prty)

# Load dataset into the memory

all_calls = pd.DataFrame(pd.read_csv('../input/911_calls_for_service.csv',index_col=0))
# Plot number of calls per month per priority group to see the seasonality of calls

call_types = pd.DataFrame([['All','Medium'],['High','Low']])



fig, axarr = plt.subplots(2,2)



for i in range(2):

    for j in range(2):

        for yr in [2015,2016,2017]:

            # print count_it(all_calls,yr)

            if yr == 2017:

                axarr[i,j].plot(pd.Series(range(1,8)),

                        count_it(all_calls,yr,prty=call_types[i][j],weekly=False),

                        label = yr,marker = '.')

            else:

                axarr[i,j].plot(pd.Series(range(1,13)),

                        count_it(all_calls,yr,prty=call_types[i][j],weekly=False),

                        label = yr,marker = '.')



            # if call_types[i][j] == 'Low': axarr[i,j].legend(loc='upper right')

            axarr[i,j].set_xlim(xmin=1, xmax=12)

            axarr[i,j].set_title(call_types[i][j])



axarr[1,1].legend(loc='upper right')

plt.setp([a.set_xticks(range(1,13,2)) for a in axarr[1, :]])

plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)

plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.show()
# Calculate weekly median number of calls and run t-test

df_weekly_medians = pd.DataFrame()

df_weekly_calls_ttest = pd.DataFrame()



for priority in ['All','High','Medium','Low']:

    df_weekly_medians.insert(0,priority,value = weekly_medians(all_calls,priority))

    df_weekly_calls_ttest.insert(0,priority,value = run_ttest(all_calls,priority))

# Plot weekly median calls per year

ax = plt.subplot()

df_weekly_medians.iloc[0].plot.bar(width=0.25, color='r',position=-0.5)

df_weekly_medians.iloc[1].plot.bar(width=0.25, color='b',position=0.5)

df_weekly_medians.iloc[2].plot.bar(width=0.25, color='g',position=1.5)

ax.autoscale(tight=True)

ax.legend(loc='upper left')

plt.xticks(rotation='horizontal')

ax.set_title('Weekly Median Number of Calls per Year',fontsize = 20)

ax.set_xlabel('Call Priority')

ax.set_ylabel('Median Number of Calls')

plt.show()
# Check t-test results to see if the change in medians are statistically significant

df_weekly_calls_ttest