# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
hospital_df=pd.read_csv('/kaggle/input/hospitals-count-in-india-statewise/Hospitals count in India - Statewise.csv')
pop_df=pd.read_csv('/kaggle/input/census/census.csv')
pop_df=pop_df.iloc[:,1:]
print(hospital_df.shape)
print(pop_df.shape)

print(hospital_df.columns)

hospital_df.rename({'States/UTs':'State', 'Number of hospitals in public sector': 'Public Hospitals',
       'Number of hospitals in private sector':'Private Hospitals',
       'Total number of hospitals (public+private)':'Total Hospitals'}, axis=1, inplace=True)
#hospital_df.sort_values(by='State',inplace=True)
#pop_df.sort_values(by='State', inplace=True)
#print(hospital_df)
hospital_df['Public Hospitals']=hospital_df['Public Hospitals'].str.replace(',','')
hospital_df['Private Hospitals']=hospital_df['Private Hospitals'].str.replace(',','')

hospital_df['Total Hospitals']=hospital_df['Total Hospitals'].str.replace(',','')

hospital_df.fillna(0, inplace=True)

hospital_df['Private Hospitals']=hospital_df['Private Hospitals'].astype(int)

hospital_df['Total Hospitals']=hospital_df['Total Hospitals'].astype(int)

hospital_df['Public Hospitals']=hospital_df['Public Hospitals'].astype(int)

hospital_df.sort_values(by='State',inplace=True)
hospital_df.head()
hospital_df=hospital_df.replace({"Andaman Nicobar Islands":"Andaman and Nicobar Islands","Dadra & N Haveli":"Dadra and Nagar Haveli and Daman and Diu","Himachal Pradesh 8":'Himachal Pradesh',"Jammu & Kashmir":"Jammu and Kashmir","Daman & Diu":"Dadra and Nagar Haveli and Daman and Diu"})
hospital_df.sort_values(by='State', inplace=True)
hospital_df=hospital_df.groupby(['State']).agg('sum')
pop_df=pop_df[pop_df['State']!='India']
pop_df.replace({"Manipur[d]":"Manipur"},inplace=True)
pop_df.set_index('State', inplace=True)
print(pop_df.head())
print(hospital_df.shape)
print(pop_df.shape)
hospital_df['Population']=pop_df['Population']

hospital_df['Private Hospitals/lakh']=hospital_df.apply(lambda row: 100000*row['Private Hospitals']/row['Population'],axis=1)
hospital_df['Public Hospitals/lakh']=hospital_df.apply(lambda row: 100000*row['Public Hospitals']/row['Population'],axis=1)
hospital_df['Total Hospitals/lakh']=hospital_df.apply(lambda row: 100000*row['Total Hospitals']/row['Population'],axis=1)
hospital_df.head()
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
plt.figure(figsize=(20,6))
fig =px.pie(hospital_df, values='Total Hospitals', names=hospital_df.index)
fig.show()
plt.figure(figsize=(20,6))
fig =px.pie(hospital_df, values='Private Hospitals', names=hospital_df.index)
fig.show()
plt.figure(figsize=(20,6))
fig =px.pie(hospital_df, values='Public Hospitals', names=hospital_df.index)
fig.show()
plt.figure(figsize=(20,6))
G=sns.barplot(data=hospital_df, x=hospital_df.index, y=hospital_df['Total Hospitals/lakh'])
G.set_xticklabels(G.get_xticklabels(),rotation=90)
plt.show()

plt.figure(figsize=(20,6))
G=sns.barplot(data=hospital_df, x=hospital_df.index, y=hospital_df['Public Hospitals/lakh'])
G.set_xticklabels(G.get_xticklabels(),rotation=90)
plt.show()
plt.figure(figsize=(20,6))
G=sns.barplot(data=hospital_df, x=hospital_df.index, y=hospital_df['Private Hospitals/lakh'])
G.set_xticklabels(G.get_xticklabels(),rotation=90)
plt.show()




