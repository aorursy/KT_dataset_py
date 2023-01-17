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
dataframe = pd.read_csv('/kaggle/input/uncover/UNCOVER/covid_tracker_canada/covid-19-tracker-canada.csv')
dataframe.head()
import matplotlib.pyplot as plt
%matplotlib inline
# First, lets remove presumptive cases, so we only want to
# analyze confirmed cases

removal = dataframe[ dataframe['confirmed_presumptive'] == 'PRESUMPTIVE' ].index

#Delete these entries from the dataframe
dataframe.drop(removal , inplace=True)
# Most affected patients with travel history
travelhistory = dataframe.groupby(['travel_history']).size().sort_values(ascending=False)
pd.set_option('display.max_rows', travelhistory.shape[0]+1)
print(travelhistory.head(30))
# What province has the most cases currently (2020/03/31) ?
dataframe['province'].value_counts()
# What city has the most cases currently (2020/03/31) ?
dataframe['city'].value_counts()
# Now we analyze the patients with reported age

# We focus on the biggest available subgroup with precise entries,
# therefore remove all other "unspecified" entries

removal = dataframe[ dataframe['age'] == 'Pending' ].index

#Delete these entries from the dataframe
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'NOT YET SPECIFIED' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'Adult (not further specified)' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'Under 1' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'Between 30 and mid-70s' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'Not Provided' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'NOT SPECIFIED' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'Not Specified' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'NOT YET RELEASED' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'Teens' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'Under 10' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'under 18' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'under 20' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == '55-70' ].index
dataframe.drop(removal , inplace=True)

removal = dataframe[ dataframe['age'] == 'under 10' ].index
dataframe.drop(removal , inplace=True)
# Most affected patients by age (only entries with reported age)
dataframe['age'].value_counts()