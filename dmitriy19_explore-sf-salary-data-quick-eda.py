import pandas as pd
import seaborn as sns
import numpy as np

from collections import Counter

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/Salaries.csv')
data.head(3)
data.info()
series_list = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']
for series in series_list:
    data[series] = pd.to_numeric(data[series], errors='coerce')
data['Year'].unique()
print('Agency unique values:')
data['Agency'].unique()
print('Status unique values:')
data['Status'].unique()
data.drop(['Notes', 'Agency'], axis=1, inplace=True)
data.describe()
data[data['TotalPay'] < 0]
data[data['TotalPay'] == 0].head(3)
len(data[data['TotalPay'] == 0])
data[(data['TotalPay'] > 0) & (data['TotalPay'] <= 400)].head(3)
len(data[(data['TotalPay'] > 0) & (data['TotalPay'] < 400)])
g = sns.FacetGrid(data, col="Year", col_wrap=2, size=5, dropna=True)
g.map(sns.kdeplot, 'TotalPay', shade=True);
ft = data[data['Status'] == 'FT']
pt = data[data['Status'] == 'PT']

fig, ax = plt.subplots(figsize=(9, 6))

sns.kdeplot(ft['TotalPay'].dropna(), label="Full-Time", shade=True, ax=ax)
sns.kdeplot(pt['TotalPay'].dropna(), label="Part-Time", shade=True, ax=ax)

plt.xlabel('Total Pay')
plt.ylabel('Density')
title = plt.title('Total Pay Distribution')
fig, ax = plt.subplots(figsize=(9, 6))

sns.kdeplot(ft['BasePay'].dropna(), label="Full-Time", shade=True, ax=ax)
sns.kdeplot(pt['BasePay'].dropna(), label="Part-Time", shade=True, ax=ax)

plt.xlabel('Base Pay')
plt.ylabel('Density')
title = plt.title('Base Pay Distribution')
fig, ax = plt.subplots(figsize=(9, 6))

sns.kdeplot(ft['OvertimePay'].dropna(), label="Full-Time", shade=True, ax=ax)
sns.kdeplot(pt['OvertimePay'].dropna(), label="Part-Time", shade=True, ax=ax)

plt.xlabel('Overtime Pay')
plt.ylabel('Density')
title = plt.title('OvertimePay Distribution')
fig, ax = plt.subplots(figsize=(9, 6))

sns.kdeplot(ft['Benefits'].dropna(), label="Full-Time", shade=True, ax=ax)
sns.kdeplot(pt['Benefits'].dropna(), label="Part-Time", shade=True, ax=ax)

plt.xlabel('Benefits')
plt.ylabel('Density')
title = plt.title('Benefits Distribution')
print('All unique job titles:', len(data['JobTitle'].unique()) - 1)
print('Full-time unique job titles:', len(ft['JobTitle'].unique()) - 1)
print('Part-time unique job titles:', len(pt['JobTitle'].unique()) - 1)
from collections import Counter
job_titles = data['JobTitle'].unique()[:-1] # deleting the last element "Not provided"

words_in_titles = []

for job_title in job_titles:
    words_in_titles += job_title.lower().split()
    
# a little cleaning
words = []
for word in words_in_titles:
    if not word.isdigit() and len(word) > 3:
        words.append(word)
    
words_count = Counter(words)
# words_count.most_common(200)
job_groups = {'Fire'    : ['fire'],
              'Airport' : ['airport'],
              'Animal'  : ['animal'],
              'Mayor'   : ['mayor'],
              'Library' : ['librar'],
              'Parking' : ['parking'],
              'Clerk'   : ['clerk'],
              'Porter'  : ['porter'],
              'Engineer and Tech': ['engineer', 'programmer', 'electronic', 'tech'], 
              'Court'   : ['court', 'legal', "attorney's", 'atty', 'eligibility'], 
              'Police'  : ['sherif', 'officer', 'police', 'probation', "sheriff's", 'sergeant'],
              'Medical' : ['nurse', 'medical', 'health', 'physician', 'therapist', 'psychiatric', 'treatment', 'hygienist'],
              'Public Works' : ['public'],
              'Food Service' : ['food'],
              'Architectural' : ['architect']}
def transform_func(title):
    title = title.lower()
    for key, value in job_groups.items():
        for each_value in value:
            if title.find(each_value) != -1:
                return key
    return 'Other'
data['JobGroup'] = data['JobTitle'].apply(transform_func)
data.head(3)
g = sns.FacetGrid(data, col="JobGroup", col_wrap=3, size=4.5, dropna=True)
res = g.map(sns.kdeplot, 'TotalPay', shade=True)
ft_med = data[(data['Status'] == 'FT') & (data['JobGroup'] == 'Medical')]
pt_med = data[(data['Status'] == 'PT') & (data['JobGroup'] == 'Medical')]
fig, ax = plt.subplots(figsize=(9, 6))

sns.kdeplot(ft['TotalPay'].dropna(), label="Full-Time", shade=True, ax=ax)
sns.kdeplot(pt['TotalPay'].dropna(), label="Part-Time", shade=True, ax=ax)

plt.xlabel('TotalPay')
plt.ylabel('Density')
title = plt.title('Medical Total Pay Distribution')