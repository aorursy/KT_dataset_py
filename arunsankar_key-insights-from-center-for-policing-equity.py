import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))
for root, dirs, files in os.walk("../input"):
    for file in files:
        print(file)
print(os.listdir("../input/cpe-data"))
acs = pd.read_csv('../input/cpe-data/ACS_variable_descriptions.csv')
acs.head()
print(os.listdir("../input/cpe-data/Dept_23-00089/"))
print(os.listdir("../input/cpe-data/Dept_35-00103/"))
print(os.listdir("../input/cpe-data/Dept_11-00091/"))
print(os.listdir("../input/cpe-data/Dept_37-00049/"))
print(os.listdir("../input/cpe-data/Dept_37-00027/"))
print(os.listdir("../input/cpe-data/Dept_49-00009/"))
d35_prep = pd.read_csv('../input/cpe-data/Dept_35-00103/35-00103_UOF-OIS-P_prepped.csv')
d35_prep.head()
print('\nPrep data: \nRows: {}\nCols: {}'.format(d35_prep.shape[0],d35_prep.shape[1]))
print(d35_prep.columns)
race_incidents = d35_prep[1:].groupby('SUBJECT_RACE').count().reset_index()[['SUBJECT_RACE','INCIDENT_UNIQUE_IDENTIFIER']].sort_values('INCIDENT_UNIQUE_IDENTIFIER', ascending=False)

fig, ax = plt.subplots(figsize=(8,6))
a = sns.barplot(x='SUBJECT_RACE', y='INCIDENT_UNIQUE_IDENTIFIER', data=race_incidents, ax=ax, color="#2196F3")
#a.set_xticklabels(labels=games_athletes['Games'],rotation=90)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

ax.set_xlabel('Race', size=14, color="#0D47A1")
ax.set_ylabel('Number of Incidents', size=14, color="#0D47A1")
ax.set_title('[BAR] Dept35 - Number of Incidents by Race', size=18, color="#0D47A1")

plt.show()
d3749_prep = pd.read_csv('../input/cpe-data/Dept_37-00049/37-00049_UOF-P_2016_prepped.csv')
d3749_prep.head()
print('\nPrep data: \nRows: {}\nCols: {}'.format(d3749_prep.shape[0],d3749_prep.shape[1]))
print(d3749_prep.columns)
d3727_prep = pd.read_csv('../input/cpe-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv')
d3727_prep.head()
print('\nPrep data: \nRows: {}\nCols: {}'.format(d3727_prep.shape[0],d3727_prep.shape[1]))
print(d3727_prep.columns)