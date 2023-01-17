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

        assessment_dataset_path = os.path.join(dirname, filename)

        assessment_data = pd.read_csv(assessment_dataset_path)



# Any results you write to the current directory are saved as output.

assessment_data.head()
assessment_data = assessment_data.rename(columns = {'Date' : 'date', 'purchase_amonut' : 'purchase_amount'})

assessment_data.head()
print('Is there any missing value in the assessment data?',

      assessment_data.isnull().values.any())
import time

import datetime



normalized_date = []

min_date = time.mktime(datetime.datetime.strptime(assessment_data.loc[0, 'date'], "%m/%d/%Y").timetuple())

max_date = time.mktime(datetime.datetime.strptime(assessment_data.loc[0, 'date'], "%m/%d/%Y").timetuple())



for i, row in assessment_data.iterrows():

    timestamp = time.mktime(datetime.datetime.strptime(row['date'], "%m/%d/%Y").timetuple())

    if min_date > timestamp:

        min_date = timestamp

    if max_date < timestamp:

        max_date = timestamp

    normalized_date.append(int(timestamp))

    

assessment_data['normalized_date'] = normalized_date

assessment_data.head()
from datetime import datetime



date_domain = max_date - min_date



for i, row in assessment_data.iterrows():

    assessment_data.loc[i, 'normalized_date'] = (int(row['normalized_date']) - min_date) / date_domain

    

assessment_data.head()
print('Starting date: ', datetime.fromtimestamp(min_date).date())

print('Ending date: ', datetime.fromtimestamp(max_date).date())



print('The number of records: ', len(assessment_data))

print('The number of unique user IDs: ', len(assessment_data['user_id'].unique()))

print('The number of unique devices: ', len(assessment_data['device'].unique()))

print('The number of unique locations: ', len(assessment_data['location'].unique()))
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



plt.figure(figsize=(18,12))

plot = sns.distplot(a=assessment_data["normalized_date"]*4*31, kde=True, color='yellow')

plot.set(xlabel ='The days from May 1 to August 31', ylabel ='The frequency of purchases')

plt.show()
plt.figure(figsize=(18,12))

plot = sns.distplot(a=assessment_data["purchase_amount"], kde=True, color='orange')

plot.set(xlabel ='The amount of purchase', ylabel ='Frequency')

plt.show()
sizes = []

devices = assessment_data['device'].unique()



for device in devices:

    sizes.append(len(assessment_data[assessment_data['device'] == device]))

    

sizes



fig, ax = plt.subplots(figsize=(18,12))



ax.pie(sizes,

       labels= devices,

      autopct='%1.1f%%')



ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
amounts = []

devices = assessment_data['device'].unique()



for device in devices:

    amounts.append(assessment_data[assessment_data['device'] == device]['purchase_amount'].sum() /

                   len(assessment_data[assessment_data['device'] == device]))



fig, ax = plt.subplots(figsize=(18,12))

    

ax.pie(amounts,

       labels= devices,

      autopct='%1.1f%%')



ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
amounts = []

locations = assessment_data['location'].unique()



for location in locations:

    amounts.append(assessment_data[assessment_data['location'] == location]['purchase_amount'].sum())



fig, ax = plt.subplots(figsize=(18,12))

    

ax.pie(amounts,

       labels= locations,

      autopct='%1.1f%%')



ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
amounts = []

locations = assessment_data['location'].unique()



for location in locations:

    amounts.append(assessment_data[assessment_data['location'] == location]['purchase_amount'].sum() /

                  len(assessment_data[assessment_data['location'] == location]))



fig, ax = plt.subplots(figsize=(18,12))

    

ax.pie(amounts,

       labels= locations,

      autopct='%1.1f%%')



ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
plt.figure(figsize=(18,12))

sns.scatterplot(x=assessment_data['location'], y=assessment_data["purchase_amount"], hue=assessment_data['user_id'], legend = 'full')

plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))

plt.show()
plt.figure(figsize=(18,12))

sns.scatterplot(x=assessment_data["normalized_date"]*31*4, y=assessment_data["purchase_amount"], hue=assessment_data["location"])

plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))

plot.set(xlabel ='Day', ylabel ='The amount of purchase')

plt.show()

plt.figure(figsize=(18,12))

sns.scatterplot(x=assessment_data['user_id'], y=assessment_data["purchase_amount"], hue=assessment_data['device'], palette = 'Set2')

plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))

plt.show()
amounts = []

users = assessment_data['user_id'].unique()



for user in users:

    amounts.append(assessment_data[assessment_data['user_id'] == user]['purchase_amount'].sum() /

                   len(assessment_data[assessment_data['user_id'] == user]))



fig, ax = plt.subplots(figsize=(18,12))

    

ax.pie(amounts,

       labels= users,

      autopct='%1.1f%%')



ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
plt.figure(figsize=(18,12))



for i in range(len(assessment_data['user_id'].unique())):

    user_id = assessment_data['user_id'].unique()[i]

    sns.lineplot(x=assessment_data[assessment_data['user_id'] == user_id]['normalized_date']*31*4, y='purchase_amount', data=assessment_data[assessment_data['user_id'] == user_id], legend='brief')

plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))

plt.show()

# plot = sns.distplot(a=assessment_data["normalized_date"]*4*31, kde=True, color='yellow')

# plot.set(xlabel ='The days from May 1 to August 31', ylabel ='The frequency of purchases')

# plt.show()
print('Average purchase amount: ', assessment_data['purchase_amount'].sum() / len(assessment_data))
print('Average sales amount per day: ', assessment_data['purchase_amount'].sum() / (31*4))

print('Average sales amount per week: ', assessment_data['purchase_amount'].sum() / (31*4/7))

print('Average sales amount per month: ', assessment_data['purchase_amount'].sum() / 4)
print('Average sales amount per user: ', assessment_data['purchase_amount'].sum() / len(assessment_data['user_id'].unique()))



amounts = []

users = assessment_data['user_id'].unique()



for user in users:

    amounts.append(assessment_data[assessment_data['user_id'] == user]['purchase_amount'].sum())



print('Minimum of total amount of purchases of users: ', np.min(amounts))    

print('Median of total amount of purchases of users: ', np.median(amounts))

print('Maximum of total amount of purchases of users: ', np.max(amounts))