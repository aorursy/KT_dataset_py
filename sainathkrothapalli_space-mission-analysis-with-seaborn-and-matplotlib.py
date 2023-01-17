# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')

data.head()
data.shape
data.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1,inplace=True)
data.info()
data['Status Mission'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(data['Status Mission'])
data['Status Mission'].value_counts().plot.pie(autopct='%1.1f%%',figsize=(8,8))
data['Company Name'].value_counts()[:15]
data['Company Name'].value_counts()[:15].plot(kind='bar')
data['Status Rocket'].value_counts()
sns.countplot(data['Status Rocket'])
sns.countplot(data['Status Mission'],hue='Status Rocket',data=data)
success=data[data['Status Mission']=='Success']

success['Company Name'].value_counts()[:15].plot(kind='bar',color='red')

sns.countplot(success['Status Rocket'])
Failure=data[data['Status Mission']=='Failure']

Failure['Company Name'].value_counts()[:15].plot(kind='bar',color='green')
sns.countplot(Failure['Status Rocket'])
def get_year(x):

    return x[12:16]

data['Year']=data['Datum'].map(get_year)

data['Year']=data['Year'].astype('int64')

data['Year'].value_counts()[:10].plot(kind='bar',color='brown')
Success=data[data['Status Mission']=='Success']

Success['Year'].value_counts()[:10].plot(kind='bar',color='brown')
def get_month(x):

    return x[4:7]

data['Month']=data['Datum'].map(get_month)

sns.countplot(data['Month'])
Successs=data[data['Status Mission']=='Success']

Successs['Month'].value_counts()[:10].plot(kind='bar',color='yellow')
spacex=data[data['Company Name']=='SpaceX']

print("No Of rockets launched by Spacex",spacex.shape[0])

sns.countplot(spacex['Status Mission'])
sns.countplot(spacex['Status Rocket'])
data['country'] = data['Location'].str.split(', ').str[-1]

data['country'].head()            
data['country'].value_counts()[:10].plot(kind='bar',color='blue')
Success1=data[data['Status Mission']=='Success']

Success1['country'].value_counts()[:10].plot(kind='bar',color='brown')