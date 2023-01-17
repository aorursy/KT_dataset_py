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
gun_df = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
gun_df.describe()
gun_df.info()
total = gun_df.isnull().sum().sort_values(ascending= False)
percent = (gun_df.isnull().sum()/gun_df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
state_max = gun_df.groupby('state').count().sort_values(ascending= False, by=['incident_id'])
state_max_new = state_max['incident_id'].head(10)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.barplot(x = state_max_new.values, y=state_max_new.index)
state_max_new
city_max = gun_df.groupby('city_or_county').count().sort_values(ascending= False, by=['incident_id']).head(10)['incident_id']
sns.barplot(x = city_max.values, y=city_max.index)
city_max
gun_involved = gun_df['n_guns_involved'].dropna().apply(lambda x : "4+" if x>4 else str(x))
gun_involved = gun_involved.groupby(gun_involved.values)
gun_involved = gun_involved.count()
gun_involved.plot(kind='pie', title='Number of guns involved', autopct='%1.0f%%', subplots=True, figsize=(8, 8))
import datetime as dt
gun_df['date'] = pd.to_datetime(gun_df['date'])
gun_df['weekday'] = gun_df['date'].dt.weekday_name
week_wise = gun_df.groupby('weekday').sum().sort_values(ascending = False, by=['incident_id'])
week_wise['dayname'] = week_wise.index
week_wise.plot(x='dayname',y=['n_killed','n_injured'],kind='bar')
gun_df['month'] = gun_df['date'].dt.month
month_dict = {1 :"Jan",2 :"Feb",3 :"Mar",4 :"Apr",5 : "May",6 : "Jun",7 : "Jul",8 :"Aug",9 :"Sep",10 :"Oct",11 :"Nov",12 :"Dec"}
gun_df['month'] = gun_df['month'].map(month_dict)
month_wise = gun_df.groupby('month').sum().sort_values(ascending = False, by=['incident_id'])
month_wise['monthname'] = month_wise.index
month_wise.plot(x='monthname',y=['n_killed','n_injured'],kind='bar')
