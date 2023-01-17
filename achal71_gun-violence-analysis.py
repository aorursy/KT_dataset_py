# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt      # For base plotting
# Seaborn is a library for making statistical graphics
# in Python. It is built on top of matplotlib and 
#  numpy and pandas data structures.
import seaborn as sns                # Easier plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
# Read data file
data = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')

data.columns
data.shape                           # dim()

data.head(1)                          # head()

data['victims']=data['n_killed']+data['n_injured']

data.shape
#Extract Year and Month from dates
data['date'] = pd.to_datetime(data['date'])      # Convert to datetime
data.dtypes

# Now create columns and extract
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data.head()
data.shape

#How many states are there and how many incidents have been reported in each state
len(data['state'].unique())         # Number of states where the incidents have been reported
data['state'].unique()              # Which states
data['state'].value_counts()        # Distribution

data['state'].value_counts()[:20]
hi_crime_states = data['state'].value_counts()[:20].index.tolist()
hi_crime_states
#bin_new = np.arange(start=5000, stop=18000, step=1000)
#data[hi_crime_states].plot(kind='hist', bins=bin_new, figsize=[12,6], alpha=.4, legend=True)

y_pos = np.arange(len(hi_crime_states))

plt.bar(y_pos, data['state'].value_counts()[:20], align='center', alpha=0.5)
plt.xticks(y_pos, hi_crime_states, rotation = 'vertical')
plt.ylabel('No. of incidents')
plt.title('States with highest number of gun violence incidents')
 
plt.show()

#Year-wise people killed or injured
dd = data.groupby(['year'])['victims'].sum()
dd[dd > 2013]
#  dd[(dd<2018)]
dd
#Year-wise number of incidents
qq = data.groupby(['year'])['incident_id'].count()
qq

#month-wise - is there any trend in number of killings?
mm = data.groupby(['month'])
mmfiltered = mm.filter(lambda x: (x['year'] != 2018).any())
mmfiltered.groupby(['month'])['victims'].sum()

#month-wise - is there any trend in number of incidents?
mmfiltered.groupby(['month'])['incident_id'].count()

xx = mmfiltered.groupby(['month'])['incident_id'].count()

y_pos = np.arange(len(xx))
plt.bar(y_pos, xx, align='center', alpha=0.5)
plt.xticks(y_pos, 'JFMAMJJASOND', rotation = 'vertical')
plt.ylabel('No. of incidents')
plt.title('Month wise trend')
plt.show()

#What is the number of people that are killed/ injured typically in each of these incidents

(data['victims']).sort_values(ascending=False)
#type((data['victims']).sort_values(ascending=False))
#len((data['victims']).sort_values(ascending=False))

#  How many victims are there in each incident ###############
data['victims'].max()

bin_values = np.arange(start=0, stop=120, step=4)
print(bin_values)
bin_values2 = np.arange(start=0, stop=10, step=1)

data['victims'].hist(bins=bin_values, figsize=[8,2])



data[data['victims']<=10]['victims'].hist(bins=bin_values2, figsize=[8,2])

data[data['victims']>10]['victims'].hist(bins=bin_values, figsize=[8,2])

# Number of incidents corresponding to number of victims in the incidents 
(data['victims']).value_counts()

ageOfSuspect = []
for row in range(0,len(data)-1):
#    print("Row number", row)    
    if(not pd.isnull(data.loc[row,'participant_age'])):
        for x in data.loc[row,'participant_type'].split('||'):
            if('Subject-Suspect' in str(x)):
#                print(str(x)[3:])
                for y in data.loc[row,'participant_age'].split('||'):
                    if(str(y)[0]==str(x)[0]):
#                        print(str(y)[3:])
                        ageOfSuspect.append(y[3:5])

ageOfSuspect
#data['suspect_age'] = ageOfSuspect
#data.head()
len(ageOfSuspect)
type(ageOfSuspect)

ageOfSuspect.count('15')    

from collections import Counter
c=Counter(ageOfSuspect)
#print(c.items())

del c['::'],c[''],c[':1'],c[':2'],c[':3'],c[':4'],c[':6'],c[':7'],c['|1'],c['1|'], c['4|'], c['2|'], c['3|'], c['5|'], c['6|'], c['8|'], c['9|'], c['7|'], c['0|']
print(c.items())

colors = list("rgbcmyk")

type(c)
key = c.keys()
df = pd.DataFrame(c,index=key)
df
df.drop(df.columns[1:], inplace=True)

type(df)
row = df.iloc[0]
row.sort_index()
row.sort_index().plot(kind='bar')



row2 = row[row > 1000]
row2.sort_index()
row2.sort_index().plot(kind='bar')

sns.boxplot(data= row)
sns.violinplot(data= row)