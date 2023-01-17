# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/ParticipatingEmployersJune2018Cleaned.csv")

nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
df.head()

plt.figure(figsize=(15,8))
total = float(len(df))
sns.countplot(y='Company Size', data = df)
csize = df.groupby(['Company Size']).size()
sizedf = pd.DataFrame()
sizedf['Company size'] = csize.index
sizedf['counts'] = csize.values
sizedf['%']  = (sizedf['counts'] /total) *100
sizedf.sort_values(by='counts',ascending=False)

states_df = df.groupby(['State']).size()

statesdf = pd.DataFrame()
statesdf['state'] = states_df.index
statesdf['counts'] = states_df.values
statesdf.sort_values(by = 'counts',ascending = False)

plt.figure(figsize=(20,30))


p1 = plt.barh(statesdf['state'], statesdf['counts'])

plt.show()

overview = pd.DataFrame()
percent = (statesdf['counts'] / total )*100
overview['states'] = statesdf['state']
overview['counts'] = statesdf['counts']
overview['percent %'] = percent

overview.sort_values(by = 'percent %',ascending = False)
plt.figure(figsize=(10,6))
sns.countplot(x='Federal Contractor?', data= df)
explode = (0.1,0)  
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df['Federal Contractor?'].value_counts(), explode=explode,labels=['No','Yes'], autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
csize_fc = df[df['Federal Contractor?']=='Yes'].groupby(['Company Size']).size()
sizedf_fc = pd.DataFrame()
sizedf_fc['Company size'] = csize_fc.index
sizedf_fc['counts'] = csize_fc.values

plt.figure(figsize=(15,8))


p1 = plt.barh(sizedf_fc['Company size'], sizedf_fc['counts'])


plt.show()
plt.figure(figsize=(9,5))
sns.countplot(x='E-Verify Employer Agent?', data= df)
