# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import matplotlib.pyplot as plt

from matplotlib import colors

from collections import defaultdict





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
f = open('/kaggle/input/bank-marketing-dataset/bank.csv')

all_lines = csv.reader(f,delimiter = ",")



dataset = []

header = next(all_lines)



#convert numeric fields to int

for line in all_lines:

    d = dict(zip(header,line))

    d['age'] = int(d['age'])

    d['duration'] = int(d['duration'])

    d['campaign'] = int(d['campaign'])

    d['pdays'] = int(d['pdays'])

    d['previous'] = int(d['previous'])

    d['balance'] = int(d['balance'])

    dataset.append(d)



dataset[0]

    
len(dataset)
#calculate average yearly balance

balances = [d['balance'] for d in dataset]

balances = np.array(balances)

print('Average Balance is : ',np.mean(balances))
print('Highest Balance  : ',np.max(balances))
print('Lowest Balance  : ',np.min(balances))
#How many clients subscribed a term deposit? 

clients = [d['deposit'] for d in dataset if d['deposit']=='yes']

clients = np.array(clients)

print('Clients who subscribed to term deposit is : ',clients.size)
# to find potential clients we identify employed who have not subscribed to fixed deposit

potentials = [d['deposit'] for d in dataset if d['deposit']=='no']

potentials = np.array(potentials)

print('Total number of potential clients: ',potentials.size)
#age disrtribution

ages = [d['age'] for d in dataset]

ages.sort()



client_ages = defaultdict(int)

for p in ages:

    client_ages[p] += 1
# Flatten distribution list into frequency distribution

age_freq = []

for key in client_ages.keys():

    for i in range(0, client_ages.get(key)):

        age_freq.append(key)



hist,bin_edges = np.histogram(age_freq)

plt.figure(figsize=[9,7])

n, bins, patches = plt.hist(x=age_freq, bins=8, color='#0504aa', alpha=0.7, rwidth=0.9)

plt.grid(axis='y',alpha=0.75)

plt.xlabel('Age', fontsize=15)

plt.ylabel('Clients', fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.title('Clients Age Distribution', fontsize=15)

plt.show()
plot_data = [age_freq]

fig = plt.figure(1, figsize=(12, 9))



# Create an axes instance

ax = fig.add_subplot(111)



top = 20

bottom = 0

#ax.set_ylim(bottom, top)

ax.set_xticklabels(['Age'])

ax.get_xaxis().tick_bottom()

ax.get_yaxis().tick_left()

# Create the boxplot

bp = ax.boxplot(plot_data, patch_artist=True)



plt.title('Distributions of Age')

plt.setp(bp['boxes'], color='#87deed')

plt.setp(bp['whiskers'], color='black')

plt.setp(bp['fliers'], color='red', marker='.')

plt.show()
# Stacked Bar Chart showing clients who subscribed vs those who didnt

subscribed = np.array([d['deposit'] for d in dataset if d['deposit'] == 'yes']).size

notsubscribed = np.array([d['deposit'] for d in dataset if d['deposit'] == 'no']).size

index = [1]



print(subscribed)



p1 = plt.bar(index, subscribed, color='lightblue')

p2 = plt.bar(index, notsubscribed, bottom=subscribed, color='pink')

plt.gca().set(title='Clients by Subscription Status', ylabel='Clients');

plt.xticks([])



plt.legend((p1[0], p2[0]), ('Subscribed', 'Not Subscribed'))

plt.show()
#correlation between Job and subscription

admin = np.array([d['deposit'] for d in dataset if d['job']=='admin' and d['deposit']=='yes']).size

unknown = np.array([d['deposit'] for d in dataset if d['job']=='unknown' and d['deposit']=='yes']).size

unemployed = np.array([d['deposit'] for d in dataset if d['job']=='unemployed' and d['deposit']=='yes']).size

management = np.array([d['deposit'] for d in dataset if d['job']=='management' and d['deposit']=='yes']).size

housemaid = np.array([d['deposit'] for d in dataset if d['job']=='housemaid' and d['deposit']=='yes']).size

entrepreneur = np.array([d['deposit'] for d in dataset if d['job']=='entrepreneur' and d['deposit']=='yes']).size

student = np.array([d['deposit'] for d in dataset if d['job']=='student' and d['deposit']=='yes']).size

bluecollar = np.array([d['deposit'] for d in dataset if d['job']=='blue-collar' and d['deposit']=='yes']).size

selfemployed = np.array([d['deposit'] for d in dataset if d['job']=='self-employed' and d['deposit']=='yes']).size

retired = np.array([d['deposit'] for d in dataset if d['job']=='retired' and d['deposit']=='yes']).size

technician = np.array([d['deposit'] for d in dataset if d['job']=='technician' and d['deposit']=='yes']).size

services = np.array([d['deposit'] for d in dataset if d['job']=='services' and d['deposit']=='yes']).size



jobs = np.array(['Admin','Unknown','Unemployed','Management','Housemaid','Entrepreneur','Student','Blue-collar','Self-employed',

                'Retired','Technician','Services'])



subscribed = np.array([admin,unknown,unemployed,management,housemaid,entrepreneur,student,bluecollar,selfemployed,

                retired,technician,services])







plt.figure(figsize=[12,9])

plt.xlabel('Job')

plt.ylabel('Clients')

plt.title('Correlation of Job and Subscription to term deposit.')

plt.scatter(jobs,subscribed)

plt.xticks(rotation=60)

plt.show()