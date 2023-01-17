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

import seaborn as sns

import matplotlib.pyplot as plt

import re
corona_cleaned = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")

corona_summary = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126 - SUMMARY.csv")









name = os.listdir('/kaggle/input/2019-coronavirus-dataset-01212020-01262020')

address = list()

for x in name:

    if(re.search("CLEANED.csv$|Cleaned.csv$|cleaned.csv$|SUMMARY.csv$|Summary.csv$|summary.csv$",x)== None):

        address.append(x)

address


datafull = pd.DataFrame()

for x in address:

    df = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/'+x,parse_dates=['Last Update'])

    datafull = pd.concat([datafull,df])
datafull
#There are duplicates in our data
datafull.drop_duplicates(inplace = True)
datafull.sort_values(by = ['Province/State','Last Update'],ascending=False,inplace=True)
datafull
#Making Last update as index in a new dataframe data_series
data_series =datafull.set_index('Last Update')
#Converting Missing values in provinces into "Unknown"
data_series['Province/State'].fillna('Unknown',inplace=True)
#Making data into bins of 1 day each instead of having multiple values in each day
group = data_series.groupby('Province/State')
time_series = pd.DataFrame()

for name in data_series['Province/State'].unique():

    time_series = pd.concat([time_series,group.get_group(name).resample('1D').max()])
#Treating Missing Values
time_series["Province/State"] = time_series["Province/State"].fillna("Unknown")

cols = ['Confirmed','Suspected','Recovered','Death']

for col in cols:

    time_series[col].fillna(value = 0,inplace = True)
time_series.isna().sum()
#This missing data left is basically completely empty rows 
time_series[time_series["Country/Region"].isna()==True]
#We will drop

time_series.dropna(inplace=True)
time_series