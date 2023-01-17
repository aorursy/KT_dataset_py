import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

import matplotlib.pyplot as plt  

import seaborn as sns

ttc_data = pd.read_csv('../input/to-subway-delay/may.csv',index_col='Date',parse_dates=True)

# This step is to drop all the rows has null/ n/a data

ttc_data = ttc_data.dropna(axis=0)

#Load the first 10 rows of data from the dataset

print(ttc_data.head(10))
sns.set(style='darkgrid')

delay_station = sns.countplot(y='Station',data =ttc_data,order=ttc_data['Station'].value_counts().iloc[:5].index)

sns.set(style='darkgrid')

delay_day = sns.countplot(y='Day',data =ttc_data,order=ttc_data['Day'].value_counts().iloc[:7].index)


sns.catplot(x="Day", y="Min Delay",data=ttc_data);
sns.set(style='darkgrid')

delay_bound = sns.countplot(y='Bound',data =ttc_data,order=ttc_data['Bound'].value_counts().iloc[:4].index)
sns.set(style='darkgrid')

delay_type = sns.countplot(y='Code',data =ttc_data,order=ttc_data['Code'].value_counts().iloc[:10].index)

t_codes = ['SUDP','TUSC','MUPAA']

g = sns.catplot(x='Code',col='Station',col_wrap=5,data=ttc_data.loc[ttc_data['Code'].isin(t_codes)],kind='count',height=5,aspect=.9)