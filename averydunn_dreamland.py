# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
opioids = pd.read_csv('/kaggle/input/us-opiate-prescriptions/opioids.csv')
opioids
overdoses = pd.read_csv('/kaggle/input/us-opiate-prescriptions/overdoses.csv')
overdoses
prescribers = pd.read_csv('/kaggle/input/us-opiate-prescriptions/prescriber-info.csv')
prescribers.head(5)
# Did more male or female practitioners prescribe opioids?
prescribers['Gender'].value_counts()
# Which specialty of medicine prescribed opioids the most?
prescribers['Specialty'].value_counts()
# Which state had the highest number of prescribers prescribing opioids?
prescribers['State'].value_counts()
AL = prescribers[prescribers['State'] == 'AL']

# Labels whether or not individual prescribed opioids more than 10 times in the year
AL = AL.drop(['NPI'], axis=1)
AL['Totals'] = AL.sum(axis=1)
AL

AL['Opioid.Prescriber'].value_counts()
AL_Totals = AL['Totals'].sum()
overdoses[overdoses['Abbrev']=='AL']
AL_overdoses = overdoses[overdoses['Abbrev']=='AL']
print('Percentage of people who have overdosed out of those prescribed opoioids in Alabama: ', 
      (round((int(AL_overdoses['Deaths'])/911474)*100, 2)))
o_list = []
for i in overdoses.Deaths: 
    i = int(i.replace(',', ''))
    o_list.append(i)
new = pd.DataFrame(o_list, columns=['Deaths'])
overdoses.update(new)
overdoses


overdoses['Deaths'].max()
overdoses[overdoses['Deaths']==4521]
CA = prescribers[prescribers['State']=='CA']
CA = CA.drop(['NPI'], axis=1)
CA['Totals'] = CA.sum(axis=1)
CA
CA['Opioid.Prescriber'].value_counts()
CA['Totals'].sum()
CA_overdoses = overdoses[overdoses['Abbrev']=='CA']
print('Percentage of people who have overdosed out of those prescribed opoioids in California: ', 
      (round((int(CA_overdoses['Deaths'])/3127430)*100, 2)))
pre_list = []
per_list = []

def percentCalc(state): 
    state1 = prescribers[prescribers['State']==state]
    state1 = state1.drop(['NPI'], axis=1)
    state1['Totals'] = state1.sum(axis=1)
    p_sum = state1['Totals'].sum()
    pre_list.append(p_sum)
    state_overdoses = overdoses[overdoses['Abbrev']==state]
    percentage = round((int(state_overdoses['Deaths'])/p_sum)*100, 2)
    per_list.append(percentage)

for state in overdoses['Abbrev']: 
    percentCalc(state)

overdoses['Prescriptions'] = pre_list
overdoses['Percentages'] = per_list

overdoses
overdoses['Percentages'].max()
overdoses[overdoses['Percentages'] == 0.77]
plt.figure(figsize=(12, 8))
sns.set()
sns.scatterplot(x='Prescriptions', y=overdoses['Deaths'], size='Population', data=overdoses, 
                legend=False)

# def label_point(x, y, val, ax):
#     a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
#     for i, point in a.iterrows():
#         ax.text(point['x']+.02, point['y'], str(point['val']))

# label_point(overdoses.Prescriptions, overdoses.Deaths, overdoses.State, plt.gca())