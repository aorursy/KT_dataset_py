# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.Series([10,20,30,40,50,60,70], index = ['A','B','C','D','E','F','G'])
data
print(data)  
print(data['A':'D'])  # This is Explicit method which including final index
print(data[1: 4])  # this is implicit method which excluding final index and it begins with 0 integer index
print(data)
print(data['A':'D'])    # It is equivalent to the explicit method loc
print(data.loc['A':'D']) # it is explicit method 
print(data.iloc[0:4])    # it is implicit method which excluding final index E 
states_capt = pd.Series({'Uttar Pradesh':'Lucknow','Madhya Pradesh': 'Bhopal','Tamil Nadu':'Chennai',
                        'Kerala':'Thiruvandanpuram','Himachal Pradesh':'Shimla' })
states_lan = pd.Series({'Uttar Pradesh':'Awadhi','Madhya Pradesh':'Hindi','Tamil Nadu':'Tamil',
                       'Kerala':'Malyalam','Himachal Pradesh':'Pahadi'})
states_d = pd.DataFrame({'Capitals':states_capt,'Language':states_lan})
states_d
states_d['Capitals']
states_d.Capitals
states_d.keys()
states_d.values
states_d.index
states_d['Uttar Pradesh': 'Kerala'] # this is the explicit method 
states_d[1:3] # This is implicit method
print(states_d)
print(states_d.loc[:'Kerala',:]) # This is explicit method   
print(states_d)
print(states_d.iloc[0:3,:]) # it is immplicit method 

