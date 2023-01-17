# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pylab as plt
from pylab import rcParams
import seaborn as sb
import csv

# Clean headers of Pandas DataFrame

proc_data = pd.read_csv('../input/procurement-notices.csv')
proc_data_df = pd.DataFrame(proc_data)
proc_data_df.columns = proc_data_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# Update publication_date datetime format

proc_data_df['publication_date'] = pd.to_datetime(proc_data_df.publication_date)
proc_data_df['deadline_date'] = pd.to_datetime(proc_data_df.deadline_date)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# number of calls currently out
# cells with NaN deadline are currently out
proc_data_df.head()
# distribution by country
# country_count = proc_data_df[['id','country_name']].groupby(['country_name']).agg(['count'])
# country_count.plot(kind='bar')
# proc_data_df.groupby('country_name').size().plot(kind='bar')

%matplotlib inline
rcParams['figure.figsize'] = 24, 22
sb.set_style('whitegrid')

plot = sb.countplot('country_name', data=proc_data_df, order=proc_data_df['country_name'].value_counts().index)
plt.xticks(rotation=90)
plt.show()
# distribution of due dates

%matplotlib inline
rcParams['figure.figsize'] = 24, 22
sb.set_style('whitegrid')

plot = sb.countplot('deadline_date', data=proc_data_df)
plt.xticks(rotation=90)
plt.show()
