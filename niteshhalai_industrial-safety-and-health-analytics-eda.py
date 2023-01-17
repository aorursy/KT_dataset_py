import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



data = pd.read_csv('/kaggle/input/industrial-safety-and-health-analytics-database/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')
data.profile_report()
data.head(10)
print('There are '+str(data.shape[0])+'rows and '+str(data.shape[1])+' column in the dataset')

print('')

print('')

print('Columns in the dataset:')

for column in data.columns:

    print (column)

    

print('')

print('')

print('Brief information about the dataset:')

print(data.info())
data.drop(labels='Unnamed_0',axis=1,inplace=True)

data.rename(columns={"Genre": "Gender", "Data": "Date"}, inplace=True)

data['Date']= pd.to_datetime(data['Date']) 

data.head(1)
data.info()
f, axes = plt.subplots(1,1,figsize=(15,5))

sns.lineplot(data = data['Date'].value_counts())
f, axes = plt.subplots(3,1,figsize=(15,5),sharex=True,sharey=True)

ax=0

for country in data['Countries'].unique():

    sns.lineplot(data = data[data['Countries']==country]['Date'].value_counts(),ax = axes[ax])

#    ax[0].set_title(country)

    ax=ax+1
#g = sns.FacetGrid(data, row='Countries')#row="smoker"

#g.map(sns.lineplot(data = data,x='date',y=data['Countries'].value_counts()))