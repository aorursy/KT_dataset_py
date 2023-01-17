%matplotlib inline
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
print(plt.style.available)
plt.style.use('ggplot')
df = DataFrame()
df = pd.read_csv("../input/crisis-data.csv")
print(list(df.columns.values))
print(df.groupby('Reported Date').count())
print(df['Precinct'].value_counts())
print(df['Precinct'].isnull().value_counts())
print(df['Initial Call Type'].isnull().value_counts())
df = df[(df['Reported Date'] != '1900-01-01') & (df['Precinct'] != 'UNKNOWN')]
df = df.dropna(subset=['Precinct','Reported Date','Initial Call Type'], how='any')
print('1900-01-01' in df['Reported Date'])
print('UNKNOWN' in df['Precinct'])
print(df['Precinct'].isnull().value_counts())
print(df['Initial Call Type'].isnull().value_counts())
print(df['Initial Call Type'].value_counts().head())
filtered = df.groupby('Initial Call Type').filter(lambda x: len(x) >= 1298)
totals = filtered.groupby(['Precinct','Initial Call Type'])['Template ID'].count().reset_index(name='Count')
print(totals)
N = 5
ind = np.arange(N)
pers = (1832,1940,852,728,2183)
suic = (1622, 2567, 843, 722, 2289)
dist = (1072, 1084, 462, 267, 1355)
susp = (356, 501, 266, 143, 422)
serv = (282, 443, 177, 113, 283)
width = 0.4

plt.figure(figsize=(15,10))
p1 = plt.bar(ind, pers, width)
p2 = plt.bar(ind, suic, width, bottom = pers)
p3 = plt.bar(ind, dist, width, bottom = pers)
p4 = plt.bar(ind, susp, width, bottom = pers)
p5 = plt.bar(ind, serv, width, bottom = pers)

plt.ylabel('Number of Occurrences')
plt.title('Distribution of Initial Call Types by Precinct')
plt.xticks(ind, ('East','North','South','Southwest','West'))
plt.yticks(np.arange(0,5000,500))
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('Person in Crisis', 'Suicide', 'Disturbance', 'Suspicious Person', 'Service'))

plt.show()