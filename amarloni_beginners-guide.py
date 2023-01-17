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

import matplotlib.pyplot as plt

%matplotlib inline



data = pd.read_csv('../input/corruption-in-india/Cases_registered_under_PCA_act_and_related_sections_IPC_2013.csv')

data.head()
data.columns
data.info()
data.isna().sum()
data.describe()
State_tot_cases = data.groupby(['STATE/UT'])['Total Cases For Investigation'].sum()

#State_tot_cases.sort(how =ascending)

State_tot_cases
state_seized_prop = data.groupby(['STATE/UT'])['Value Of Property Recovered / Seized (In Rupees)'].sum()

state_seized_prop
data['YEAR'].unique()

data.groupby(['STATE/UT'])['Cases Sent Up For Trial And Also Reported For Dept. Action'].sum()
data.plot(x ='STATE/UT', y = 'Total Cases For Investigation', kind = 'bar', figsize = (10,6))

data.plot(x ='STATE/UT', y = 'Value Of Property Recovered / Seized (In Rupees)', kind = 'bar', figsize = (10,5), color = 'r')