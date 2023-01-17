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
## Import the Data Set CSV File



accidents=pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')

accidents.head()
## Finding the type of data in each column



accidents.dtypes
## Create a Pie Chart based on Accident Sources



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



labels = accidents['Source'].unique()

accidents_by_source=[]

for label in labels:

    accidents_by_source.append(accidents[accidents['Source']==label]['ID'].count())



plt.figure(figsize=(16,10))

plt.title("Accident Distribution by Source")

plt.ylabel("Accident Count")

plt.xlabel("Source")

plt.pie(accidents_by_source,labels=labels,autopct='%1.0f%%',shadow=False,startangle=0,pctdistance=1.2,labeldistance=1.4)

plt.show()

    
## Create a Visualization of accidents based on Severity



f,ax=plt.subplots(1,2,figsize=(18,8))

accidents['Severity'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0])

ax[0].set_title('Percentage Severity Distribution')

ax[0].set_ylabel('Count')

sns.countplot(accidents['Severity'],ax=ax[1],order=accidents['Severity'].value_counts().index)

ax[1].set_title('Count of Severity')



plt.show()
## Create a Visualization of accidents based on State using Matplotlib



fig,ax = plt.subplots(figsize=(16,10))



accidents['State'].value_counts().head(50).plot.bar()
## Create a Visualization of accidents based on State using Seaborn



fig,ax = plt.subplots(figsize=(16,10))

sns.countplot(accidents['State'],order=accidents['State'].value_counts().index)
## Create a Visualization of accidents based on State and Severity



severity_1_by_state = []

severity_2_by_state = []

severity_3_by_state = []

severity_4_by_state = []

for i in accidents.State.unique():

    severity_1_by_state.append(accidents[(accidents['Severity']==1)&(accidents['State']==i)].count()['ID'])

    severity_2_by_state.append(accidents[(accidents['Severity']==2)&(accidents['State']==i)].count()['ID'])

    severity_3_by_state.append(accidents[(accidents['Severity']==3)&(accidents['State']==i)].count()['ID'])

    severity_4_by_state.append(accidents[(accidents['Severity']==4)&(accidents['State']==i)].count()['ID'])

    

plt.figure(figsize=(20,15))



plt.bar(accidents.State.unique(), severity_2_by_state, label='Severity 2')

plt.bar(accidents.State.unique(), severity_3_by_state, label='Severity 3')

plt.bar(accidents.State.unique(), severity_4_by_state, label='Severity 4')

plt.bar(accidents.State.unique(), severity_1_by_state, label='Severity 1')

plt.legend()
## Create a Visualization of top cities with the maximum number of accidents using Matplotlib



fig,ax = plt.subplots(figsize=(16,10))



accidents['City'].value_counts().head(20).plot.bar()
## What were the weather conditions of the accidents where severity is the highest?



Weather = accidents.Weather_Condition.value_counts()

severity_1_by_Weather = []

severity_2_by_Weather = []

severity_3_by_Weather = []

severity_4_by_Weather = []

for i in Weather.index:

    severity_1_by_Weather.append(accidents[(accidents['Severity']==1)&(accidents['Weather_Condition']==i)].count()['ID'])

    severity_2_by_Weather.append(accidents[(accidents['Severity']==2)&(accidents['Weather_Condition']==i)].count()['ID'])

    severity_3_by_Weather.append(accidents[(accidents['Severity']==3)&(accidents['Weather_Condition']==i)].count()['ID'])

    severity_4_by_Weather.append(accidents[(accidents['Severity']==4)&(accidents['Weather_Condition']==i)].count()['ID'])



percentage_severity_1 = []

percentage_severity_2 = []

percentage_severity_3 = []

percentage_severity_4 = []

for i in range(len(severity_1_by_Weather)):

    percentage_severity_1.append((severity_1_by_Weather[i]/Weather[i])*100)

    percentage_severity_2.append((severity_2_by_Weather[i]/Weather[i])*100)

    percentage_severity_3.append((severity_3_by_Weather[i]/Weather[i])*100)

    percentage_severity_4.append((severity_4_by_Weather[i]/Weather[i])*100)



plt.figure(figsize=(200,20))



plt.bar(Weather.index, percentage_severity_2, label='Severity 2')

plt.bar(Weather.index, percentage_severity_3, label='Severity 3')

plt.bar(Weather.index, percentage_severity_4, label='Severity 4')

plt.bar(Weather.index, percentage_severity_1, label='Severity 1')

plt.legend()