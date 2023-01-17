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
df1=pd.read_csv('../input/googleplaystore.csv')
df1.head()
df2=df1[df1['Type']!='Free']
Most_expensive_app=df1[df1['Type']=='Paid'].sort_values(by='Price',ascending=False)[['App','Price']].iloc[0]
Best_rated_app=df1.sort_values(by='Rating',ascending=False)[['App','Rating']].iloc[0]
Best_rated_app_paid=df2.sort_values(by='Rating',ascending=False)[['App','Rating']].iloc[0]
Best_rated_app_free=df1[df1['Type']=='Free'].sort_values(by='Rating',ascending=False)[['App','Rating']].iloc[0]
Categories=df1['Category'].unique()
plt.figure(figsize=(25,7))
sns.barplot(x='Category',y='Rating',data=df1,estimator=np.average)
plt.xticks(rotation=90)
plt.ylabel('Average Rating')
plt.show()
    

group=df1[['Genres','Category']].groupby('Category')


Total_Categories=df1['Category'].nunique()
Total_genres=df1['Genres'].nunique()
free_apps=df1[df1['Type']=='Free']
df1['Category'].value_counts()

df1.groupby('Android Ver').count()
df1['Content Rating'].unique()
for group,frame in df1.groupby('Content Rating'):
    max1=np.max(frame['Rating'])
    print('Highest rating '+str(max1)+' for Content Rating '+group)
df1.groupby('Category')['App'].count()
plt.figure(figsize=(20,7))
sns.barplot(x=df1['Category'].value_counts().index,y=df1['Category'].value_counts().values)
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.xticks(rotation=90)
Categories
data=df1[(df1['Category']=='HEALTH_AND_FITNESS') & (df1['Type']=='Free')]
data=data.sort_values(by='Rating',ascending=False)
#data['Installs']
data=data[[len(i)>10 for i in data['Installs'].values]].head(20)
plt.figure(figsize=(20,7))
sns.barplot(x='App',y='Rating',data=data)
plt.xticks(rotation=90)
#df1.groupby('Content Rating').count().index
plt.figure(figsize=(20,7))
sns.barplot(x=df1.groupby('Content Rating').count().index,y=df1.groupby('Content Rating').count().App)
