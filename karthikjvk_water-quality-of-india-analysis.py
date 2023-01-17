# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data = pd.read_csv("../input/IndiaAffectedWaterQualityAreas.csv",encoding='latin1')
data
# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data.columns
data['State Name'].unique()
data["State Name"].value_counts()
data['Quality Parameter'].value_counts()
data.describe()
data['Quality Parameter'].groupby(data['State Name']).describe()
data.groupby("State Name").size()
import dateutil
data['date'] = data['Year'].apply(dateutil.parser.parse)
import datetime as dt
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
State_Data = data[['State Name', 'Quality Parameter']]
import sklearn
from sklearn.preprocessing import LabelEncoder
numbers = LabelEncoder()
State_Data['Quality'] = numbers.fit_transform(State_Data['Quality Parameter'])
Group1 = State_Data.groupby(['State Name','Quality Parameter','Quality']).count().reset_index()
Group1
State_Quality_Count = pd.DataFrame({'count' : State_Data.groupby( [ "State Name", "Quality","Quality Parameter"] ).size()}).reset_index()
State_Quality_Count
TAMIL_NADU   =  State_Quality_Count[State_Quality_Count["State Name"] == "TAMIL NADU"]    
ANDHRA_PRADESH = State_Quality_Count[State_Quality_Count["State Name"] == "ANDHRA PRADESH"]
KERALA = State_Quality_Count[State_Quality_Count["State Name"] == "KERALA"]
KARNATAKA = State_Quality_Count[State_Quality_Count["State Name"] == "KARNATAKA"]
GUJARAT = State_Quality_Count[State_Quality_Count["State Name"] == "GUJARAT"]
MAHARASHTRA = State_Quality_Count[State_Quality_Count["State Name"] == "MAHARASHTRA"]
TAMIL_NADU
plt.figure(figsize=(6,4))
ax = sns.barplot(x="count", y ="Quality Parameter", data = TAMIL_NADU)
ax.set(xlabel='Count')
plt.title("Water Quality Parameter In Tamil Nadu")
plt.figure(figsize=(6,4))
ax = sns.barplot(x="count", y ="Quality Parameter", data = ANDHRA_PRADESH)
ax.set(xlabel='Count')
sns.despine(left=True, bottom=True)
plt.title("Water Quality Parameter In Andhra Pradesh")
x = State_Quality_Count.groupby('State Name')
plt.rcParams['figure.figsize'] = (9.5, 6.0)
genre_count = sns.barplot(y='Quality Parameter', x='count', data=State_Quality_Count, ci=None)
plt.show()
