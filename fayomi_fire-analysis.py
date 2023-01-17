# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('../input/LFB Incident data from Jan 2017.xlsx')
df.head()
df = df[['DateOfCall','TimeOfCall','IncidentGroup','StopCodeDescription','SpecialServiceType','IncidentStationGround']]
#visualisation of incidents

sns.countplot(df['IncidentGroup'],order=df['IncidentGroup'].value_counts().index,palette='hls')
df.head(5)
#calculate percentage of incidents

total = len(df['IncidentGroup'])

fire = len(df[df['IncidentGroup'] == 'Fire'])

special = len(df[df['IncidentGroup'] == 'Special Service'])

f_alarm = len(df[df['IncidentGroup'] == 'False Alarm'])



fire_perc = (fire/total)*100

false_alarm_perc = (f_alarm/total)*100

special_perc = (special/total)*100



print('the percentage of incidents related to Fire is ', fire_perc)

print('the percentage of incidents related to Special Services are ', special_perc)

print('the percentage of incidents related to False Alarms are ', false_alarm_perc)
total = len(spec)

animals = len(df[df['SpecialServiceType'] == 'Animal assistance incidents'])



#percentage of animal assistance

(animals/total)*100
#what are the main types of false alarms

fals = df[df['IncidentGroup'] == 'False Alarm']['StopCodeDescription']

sns.countplot(fals,order=fals.value_counts().index,palette='hls')

plt.xticks(rotation=90)
plt.figure(figsize=(16,8))

sns.countplot(df['SpecialServiceType'],order=df['SpecialServiceType'].value_counts().index,palette='Blues_r')

plt.xticks(rotation=90)
plt.figure(figsize=(14,8))

sns.countplot(df['IncidentStationGround'],order=df['IncidentStationGround'].value_counts().index)

plt.xticks(rotation=90)