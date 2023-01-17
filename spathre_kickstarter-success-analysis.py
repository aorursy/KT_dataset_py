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
#price range analysis,based on launch dates, pledged pricelines, success based on number of backers and price range,
df_main=pd.read_csv('../input/ks-projects-201801.csv')
print (df_main.columns)
df_working=df_main[['name', 'category', 'main_category', 'currency', 'deadline',
       'goal', 'launched', 'pledged', 'state', 'backers', 'country']]
df_working.head()

# df_working.hist(), minimum 50 backers considered for a successful project.

df_suc=df_working[df_working['backers']>50]
df_suc=df_working[df_working['state']=='successful']
df_suc['backers'].plot.hist(xlim=(0,1500),bins=1000,ylim=(0,125000)) 
#goals vs pledged for successful projects per category
#Backers per category for successful projects

df_ped=df_suc.groupby(['main_category'])[['goal','pledged']].sum().reset_index()
df_ped2=df_suc.groupby(['main_category'])[['backers']].mean().reset_index()
# df_ped
df_ped.plot(x="main_category", y=["goal", "pledged"], kind="bar")
df_ped2.plot(x="main_category", y=["backers"], kind="bar")
df_suc['launched'] = pd.to_datetime(df_suc['launched'])
df_suc['month']=df_suc['launched'].apply(lambda x : int(x.month))
df_suc['deadline'] = pd.to_datetime(df_suc['deadline'])
df_suc.head()
#Month based frequency of companies, successful and overall

df_working['launched'] = pd.to_datetime(df_working['launched'])
df_working['deadline'] = pd.to_datetime(df_working['deadline'])
df_working['month']=df_working['launched'].apply(lambda x : int(x.month))
df_working['month'].plot.hist(bins=12) 
df_suc['month'].plot.hist(bins=12) 
df_suc['Difference'] = df_suc['deadline'].sub(df_suc['launched'], axis=0)
df_suc.head()
#Successful projects Duration
df_suc['Difference'] = df_suc['deadline']-df_suc['launched']
df_suc['Difference'].dt.components.hours
df_suc['Difference']=df_suc['Difference'].dt.days
df_suc['Difference'].plot.hist() 
