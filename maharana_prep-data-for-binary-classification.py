# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv')

df.head()
import datetime

def convert_time(data):

    return (datetime.datetime.fromtimestamp(int(data))).strftime('%Y-%m-%d %H:%M:%S')



df['datetime_converted'] = df['Timestamp'].apply(convert_time)

df.head()

df.shape
## filtering data 

df = df[(df['datetime_converted']>= '2016-10-27') & (df['datetime_converted']< '2017-10-27')]

df.shape
## convert to day level data

df[(df.index>='2017-01-01 00:00:00') & (df.index<'2017-01-0 01:02:00') ]
df['datetime_converted'] = pd.to_datetime(df['datetime_converted'])

df.dtypes

df = df.set_index('datetime_converted')

df.head()
h_df = df.resample('1H').first()

h_df.shape
h_df[(h_df.index>='2017-01-01 00:00:00') & (h_df.index<'2017-01-03 01:02:00') ]
daily_df = df.resample('1D').first()

daily_df.shape
daily_df
## calcualting one day return

df['return_d0'] = (df['Close']/df['Close'].shift(1)) - 1

df.tail()
## making a column to store return as binary variable

df['return_target'] = np.where((df['return_d0']>0),1,0)

df.tail()
df.to_csv('out.csv')