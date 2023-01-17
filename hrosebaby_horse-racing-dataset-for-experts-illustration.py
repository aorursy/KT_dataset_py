# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print ('Columns of results.csv:')
df_results = pd.read_csv('../input/results.csv') # import barrier trial data from results.csv
print(df_results.columns.values)  # columns in results

print ('\nColumns of barries.csv:')
df_barrier = pd.read_csv('../input/barrier.csv') # import barrier trial data from barrier.csv
print(df_barrier.columns.values)  # columns in barrier trials

print ('\nColumns of comments.csv:')
df_comments = pd.read_csv('../input/comments.csv') # import comments data from comments.csv
print(df_comments.columns.values)  # columns in comments

print ('\nColumns of trackwork.csv:')
df_trackwork = pd.read_csv('../input/trackwork.csv') # import track work data from trackwork.csv
print(df_trackwork.columns.values)  # columns in track work

print ('\nColumns of horseinfo.csv:')
df_horseinfo = pd.read_csv('../input/horse_info.csv') # import horse information data from horseinfo.csv
print(df_horseinfo.columns.values)  # columns in horse information

print ('List of results dates:')
list_results_dates = (sorted(set(df_results['date']))) # list of race dates.
print (list_results_dates)

df_results.query("date=='2016-01-09' and raceno==1").sort_values(by=['row']) # select a specific race from results.
df_barrier.query("date=='2017-07-14' and raceno==1").sort_values(by=['plc']) # select a specific race from barrier trials.
df_comments.query("date=='2016-01-09' and raceno==1").sort_values(by=['horseno']) # select a specific race from comments.
#print (df_trackwork)
df_trackwork.query("horse=='A BEAUTIFUL'") # select a specific horse from track work by horse name.
df_trackwork.query("horse_code=='T421'") # select a specific horse from track work by horse code.
df_horseinfo.query("horse=='A BEAUTIFUL(T421)'") # select a specific horse from horse informa.
df_horseinfo
