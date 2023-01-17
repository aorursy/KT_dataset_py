import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

%matplotlib inline

import seaborn as sns # data visualization

import datetime



polls = pd.read_csv('../input/presidential_polls.csv')
polls.shape
polls.head()
polls.describe()
C_vs_T_vs_J_polls = polls.loc[polls.rawpoll_johnson.notnull()]
nat_C_vs_T_vs_J_polls = C_vs_T_vs_J_polls.loc[(C_vs_T_vs_J_polls.state == 'U.S.')]
nat_clinton_raw_ts = nat_C_vs_T_vs_J_polls.groupby('createddate').apply(lambda dfx: (dfx["rawpoll_clinton"] * dfx["poll_wt"]).sum() / dfx["poll_wt"].sum())

nat_clinton_raw_ts = pd.DataFrame(data = nat_clinton_raw_ts, columns = ['poll_avg_clinton'])

# This block creates dataframe of the weighted averages of Clinton's raw poll numbers by created date



nat_trump_raw_ts = nat_C_vs_T_vs_J_polls.groupby('createddate').apply(lambda dfx: (dfx["rawpoll_trump"] * dfx["poll_wt"]).sum() / dfx["poll_wt"].sum())

nat_trump_raw_ts = pd.DataFrame(data = nat_trump_raw_ts, columns = ['poll_avg_trump'])

# This block creates dataframe of the weighted averages of Trump's raw poll numbers by created date



nat_johnson_raw_ts = nat_C_vs_T_vs_J_polls.groupby('createddate').apply(lambda dfx: (dfx["rawpoll_johnson"] * dfx["poll_wt"]).sum() / dfx["poll_wt"].sum())

nat_johnson_raw_ts = pd.DataFrame(data = nat_johnson_raw_ts, columns = ['poll_avg_johnson'])

# This block creates dataframe of the weighted averages of Trump's raw poll numbers by created date



nat_CvTvJ_raw_ts = pd.concat([nat_clinton_raw_ts, nat_trump_raw_ts, nat_johnson_raw_ts], axis=1)

nat_CvTvJ_raw_ts.reset_index(level=0, inplace=True)
sns.set_style("darkgrid")

fig, ax = plt.subplots(figsize=(20,8))

plt.plot('createddate', 'poll_avg_clinton', data=nat_CvTvJ_raw_ts, marker='', color='skyblue', linewidth=2)

plt.plot('createddate', 'poll_avg_trump', data=nat_CvTvJ_raw_ts, marker='', color='red', linewidth=2)

plt.plot('createddate', 'poll_avg_johnson', data=nat_CvTvJ_raw_ts, marker='', color='yellow', linewidth=2)

plt.legend()
weekly_nat_CvTvJ_raw_ts = nat_CvTvJ_raw_ts

weekly_nat_CvTvJ_raw_ts['createddate'] = weekly_nat_CvTvJ_raw_ts['createddate'].astype('datetime64[ns]')

weekly_nat_CvTvJ_raw_ts['date_minus_time'] = weekly_nat_CvTvJ_raw_ts["createddate"].apply( lambda df : datetime.datetime(year=df.year, month=df.month, day=df.day)) 

weekly_nat_CvTvJ_raw_ts.set_index(weekly_nat_CvTvJ_raw_ts["date_minus_time"],inplace=True)
clinton_weekly_raw_polls = weekly_nat_CvTvJ_raw_ts['poll_avg_clinton'].resample('W').mean()

clinton_weekly_raw_polls = pd.DataFrame(data = clinton_weekly_raw_polls, columns = ['poll_avg_clinton'])



trump_weekly_raw_polls = weekly_nat_CvTvJ_raw_ts['poll_avg_trump'].resample('W').mean()

trump_weekly_raw_polls = pd.DataFrame(data = trump_weekly_raw_polls, columns = ['poll_avg_trump'])



johnson_weekly_raw_polls = weekly_nat_CvTvJ_raw_ts['poll_avg_johnson'].resample('W').mean()

johnson_weekly_raw_polls = pd.DataFrame(data = johnson_weekly_raw_polls, columns = ['poll_avg_johnson'])



weekly_raw_polls = pd.concat([clinton_weekly_raw_polls, trump_weekly_raw_polls, johnson_weekly_raw_polls], axis=1)

weekly_raw_polls.reset_index(level=0, inplace=True)
sns.set_style("darkgrid")

fig, ax = plt.subplots(figsize=(20,8))

plt.plot('date_minus_time', 'poll_avg_clinton', data=weekly_raw_polls, marker='', color='skyblue', linewidth=2)

plt.plot('date_minus_time', 'poll_avg_trump', data=weekly_raw_polls, marker='', color='red', linewidth=2)

plt.plot('date_minus_time', 'poll_avg_johnson', data=weekly_raw_polls, marker='', color='yellow', linewidth=2)

plt.legend()