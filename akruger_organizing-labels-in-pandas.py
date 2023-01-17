import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

%matplotlib inline

plt.style.use('ggplot')

pd.options.display.max_rows = 20
# Read in labels

base_dir = os.path.join('..', 'input')

unsorted_df = pd.read_csv(os.path.join(base_dir, 'stage1_labels.csv'))



# Get IDs for rows

s = list(range(0,len(unsorted_df),17))

obs = unsorted_df.loc[s,'Id'].str.split('_')

scanID = [x[0] for x in obs]



# Put zones in columns

columns = sorted(['Zone'+str(i) for i in range(1,18)])



df = pd.DataFrame(index=scanID, columns=columns)



# Sort labels by zone

for i in range(17):

    s = list(range(i,len(unsorted_df),17))

    df.iloc[:,i] = unsorted_df.iloc[s,1].values



print('Number of labeled scans:', len(df))

df.head()
nobj_zone = df.sum()

print(nobj_zone)



nobj_zone.plot(kind='bar', width=.75, title='Threat Count in Each Zone')

plt.ylabel('Number of Threats')

plt.xlabel('Zone')

plt.show()
nobj_scan = df.sum(1).value_counts().sort_values()

print(nobj_scan)



nobj_scan.plot(kind='bar', width=.75, title='Frequency of Threat Counts')

plt.ylabel('Number of Scans')

plt.xlabel('Threat Count')

plt.show()
df[df['Zone1']==1]
df.corr()