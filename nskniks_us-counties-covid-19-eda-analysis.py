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
# importing required library

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt  
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns  #importing packages for visualization
# Setting up the datasets
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv') #loading csv file 
df.info()

df.head()
#parasing the Date column
df['date'] =pd.to_datetime(df['date'])
df.info()
#finding data gaps:
sns.heatmap(df[['fips','cases','deaths']])

#now we need to fil the data where na
df.isnull().sum()
df.fips.fillna(df.fips.mean(),inplace=True)
df.isnull().sum()
#ADDING NEW COLUMN FOR DAY MONTH SEPETATION
t = df['date'].iloc[0]
df['Month'] = df['date'].apply(lambda t: t.month)
df['Day of Week'] = df['date'].apply(lambda t: t.dayofweek)
dmon = {1:'Jan',2:'Feb',3:'Mar',4:'Apr'}
df['Month'] = df['Month'].map(dmon)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)

df

#Top 10 county having issue 
df['county'].value_counts().head(10)
#Top 10 state having issue 
df['state'].value_counts().head(10)

#max death counts
df['deaths'].value_counts().head(5)
print('Max deaths',df['deaths'].max())
print('Min deaths',df['deaths'].min())
print('Avg deaths',df['deaths'].mean())
#max  cases
df['cases'].value_counts().head(5)
print('Max Cases',df['cases'].max())
print('Min Cases',df['cases'].min())
print('Avg Cases',df['cases'].mean())
# cases Vs Deaths 

sns.set_style('whitegrid')
sns.jointplot(x='cases',y='deaths',data=df)
   
#finding corr

sns.heatmap(df.corr())
#where is max death reported 
df[df['deaths']== df['deaths'].max()] 
#where is max death reported 
df[df['cases']== df['cases'].max()]
# Death rate month wise report
sns.barplot(x='Month',y='deaths',data=df,palette='rainbow')
# cases rate month wise report
sns.boxplot(x='Month',y='cases',data=df,palette='rainbow')
# cases rate month wise report
plt.figure(figsize =(10,3))
sns.barplot(x='state',y='deaths',data=df,palette='coolwarm')
sns.countplot(x='Month',data=df,palette='coolwarm')
