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
import matplotlib.pyplot as plt
%matplotlib inline
calls_csv = '../input/fire-department-calls-for-service.csv'
metadata_json = '../input/socrata_metadata.json'
datadict_xl = '../input/FIR-0002_DataDictionary_fire-calls-for-service.xlsx'

calls = pd.read_csv(calls_csv)
calls.head()
calls.tail()  #see last entries to figure out time period
calls.info()
calls.describe()
print(f'Number of columns: {len(calls.columns)}')
type_calls_df = calls.groupby('Call Type')['Call Number'].nunique()  
print(type_calls_df)

type_calls_df.plot(kind='bar' , color = 'blue')
plt.show()
# Counts and sorts the Call Types and creates the percentage of them 
call_pct = calls['Call Type'].value_counts(normalize=True)
call_pct
ax = call_pct.plot(kind='bar' , color = 'blue') # The majority of the calls were medical responses
ax.set_ylabel('Percentage of All Calls')
