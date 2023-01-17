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
#Day 1 of 5 Day Challenge

# Read in data and display the head to see the data

df = pd.read_csv('../input/party_in_nyc.csv')

df.head()
#The describe() function gives the count, mean and other such statistics of the data

df.describe()
df_loc = pd.read_csv('../input/bar_locations.csv')

df_loc.head()
df_loc.describe()
#Day 2 of 5 Day Challenge

# Let's do some visualizations for the data using the Seaborn library

import seaborn as sns
# Let's plot the Histogram for a numeric column in the data set such as num_calls

sns.distplot(df_loc['num_calls'],kde=False, bins=10).set_title('Number of Calls')
from scipy.stats import ttest_ind

