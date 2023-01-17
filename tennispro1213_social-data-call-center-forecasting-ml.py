# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

from dateutil import parser
# dt = parser.parse("Aug 28 1999 12:00AM")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# service_req_df = pd.read_csv('../input/service-requests-received-by-the-oakland-call-center.csv')
twitter_twcs_df = pd.read_csv('../input/customer-support-on-twitter/twcs/twcs.csv')
twitter_sample_twcs_df = pd.read_csv('../input/customer-support-on-twitter/sample.csv')
service_req_df = pd.read_csv('../input/oakland-call-center-public-work-service-requests/service-requests-received-by-the-oakland-call-center.csv')
service_req_df.head()
service_req_df['REQCATEGORY'].value_counts()
sewer_drainage_df = service_req_df.loc[service_req_df['REQCATEGORY'].isin(['DRAINAGE'])]
sewer_drainage_df['DATETIMEINIT'] = sewer_drainage_df['DATETIMEINIT'].map(lambda x: str(x).split('T')[0])
sewer_drainage_df = sewer_drainage_df[sewer_drainage_df['DATETIMEINIT'].str.contains('2017-')]
sewer_drainage_df.head()
# sewer_drainage_df['REQCATEGORY'].value_counts()
sewer_drainage_df.shape
# sewer_drainage_df.head()['DATETIMEINIT'].map(lambda x: parser.parse(x.split('T')[0]))
# sewer_drainage_df.loc['DATETIMEINIT'] = sewer_drainage_df['DATETIMEINIT'].map(lambda x: parser.parse(x.split('T')[0]))
# sewer_drainage_df['DATETIMEINIT'] = sewer_drainage_df['DATETIMEINIT'].map(lambda x: str(x).split('T')[0])
# sewer_drainage_df.head()
# sewer_drainage_df.head()['DATETIMEINIT'].map(lambda x: x.split('T')[0])
',' in '12,3'
sewer_drainage_df['DATETIMEINIT'].value_counts().sort_index().plot.line()

plumbing_g_trends_df = pd.read_csv('../input/google-trends-2017-plumber-search-oakland/plumber_search_oakland_2017.csv', skiprows=1)
mrrooters_twitter_df = pd.read_csv('../input/mrrooters-twitter-daily-follower-counts/mr_rooters_daily_followers.csv')
mrrooters_twitter_df['Date'] = mrrooters_twitter_df['Date'].map(lambda x: str(x))
mrrooters_twitter_df['Total Follower'] = mrrooters_twitter_df['Total Follower'].map(lambda x: int(x))

# mrrooters_twitter_df.loc['Total Follower'] = mrrooters_twitter_df['Total Follower'].map(lambda x: int(x))
plumbing_g_trends_df.plot.line()
# plumbing_g_trends_df.dtypes
# plumbing_g_trends_df['Category: All categories'].map(lambda x: x.split(' ')[0])
# plumbing_g_trends_df.reset_index()['index'].map(lambda x: parser.parse(x))
# plumbing_g_trends_df.head()['plumber: (San Francisco-Oakland-San Jose CA)']
# mrrooters_twitter_df.loc['Total Follower'] = mrrooters_twitter_df['Total Follower'].map(lambda x: int(x))
mrrooters_twitter_df.plot.line()
# mrrooters_twitter_df.head().Date
# mrrooters_twitter_df.dtypes

