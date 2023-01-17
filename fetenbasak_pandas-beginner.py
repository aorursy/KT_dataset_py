# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/athlete_events.csv')
data.head()
data.tail()
data.info()
data.index
data.columns
data.shape
data.values
data1=data.T
data1.head()
data.sort_values(by='Age')
data.sort_values(by='Age',ascending=False)
#selecting a single column
data['Name'] #or data.Name
#selecting,slice the rows
data[0:4]
#selecting multi-axis by label
data.loc[:,['Name','City']]
#selecting label slicing, both endpoints
data.loc[0:4,['Name','Age']]
#selecting
data.iloc[3]
#selecting
data.iloc[100:102,0:5]
#selecting slice rows
data.iloc[1:7,:]
#selecting slice columns
data.iloc[:,3:7]
#copy data
data2=data.copy()
#filtering with using isin() method
data[data['City'].isin(['London','Paris'])]
#filtering 
#Age of more than 25
filter1=data[data.Age>25]
filter1
#filtering
#Age of more than 25 AND City=London
filter2=data[data['Age']>25 & data['Team']=='Taifun']
filter2
#filtering
#City=London or 
#groupby
#number of athlete for each city
data.groupby('City').size()
#groupby
#average of height and weight for each city
data.groupby('City').agg({'Height':np.mean, 'Weight':np.mean})
data.groupby(['Sex','Sport']).agg({'Height':np.mean})
#descriptive statistics
#descriptive statistics of age
data.Age.describe()
#mean
#mean of features
data.mean()
