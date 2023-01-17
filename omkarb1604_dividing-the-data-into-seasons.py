# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

raw_data=pd.read_csv('/kaggle/input/ipl/deliveries.csv') #importing data
data=raw_data.copy() #making a copy for use as we don't want to disturb the initial import
IPL_2008=data.loc[(data['match_id']>=60)&(data['match_id']<=117)]
IPL_2009=data.loc[(data['match_id']>=118)&(data['match_id']<=174)]
IPL_2010=data.loc[(data['match_id']>=175)&(data['match_id']<=234)]
IPL_2011=data.loc[(data['match_id']>=235)&(data['match_id']<=307)]
IPL_2012=data.loc[(data['match_id']>=308)&(data['match_id']<=381)]
IPL_2013=data.loc[(data['match_id']>=382)&(data['match_id']<=457)]
IPL_2014=data.loc[(data['match_id']>=458)&(data['match_id']<=517)]
IPL_2015=data.loc[(data['match_id']>=518)&(data['match_id']<=576)]
IPL_2016=data.loc[(data['match_id']>=577)&(data['match_id']<=636)]
IPL_2017=data.loc[(data['match_id']>=1)&(data['match_id']<=59)]
IPL_2018=data.loc[(data['match_id']>=7894)&(data['match_id']<=10000)]
IPL_2019=data.loc[(data['match_id']>=11137)]
years=[2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
IPL=[IPL_2008,IPL_2009,IPL_2010,IPL_2011,IPL_2012,IPL_2013,IPL_2014,IPL_2015,IPL_2016,IPL_2017,IPL_2018,IPL_2019]