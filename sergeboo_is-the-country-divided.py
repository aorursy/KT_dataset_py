# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import csv

from IPython.display import display # Allows the use of display() for DataFrames



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
ref_results = pd.read_csv('../input/referendum.csv')

ref_results.describe()

ref_results[ref_results['Area']=='Luton']

ref_results.head(5)
ref_results['Area'].sort_values().head()
print ("There are {} regions in the referendum results".format(len(ref_results)))
def non_cast(row):

    return row['Expected Ballots'] - row['Votes Cast']



df = ref_results



df['Not Cast'] = df.apply(non_cast, axis=1)

df.head(5)
def remain_pct(row):

    # returns the percentage of votes

    return float(row['Remain'] * 100) / row['Votes Cast']



def leave_pct(row):

    # returns the percentage of votes

    return float(row['Leave'] * 100) / row['Votes Cast']



df['Remain Pct'] = df.apply(remain_pct, axis=1)

df['Leave Pct'] = df.apply(leave_pct, axis=1)



df = ref_results.sort_values(by='Percent Turnout', ascending=True).head(25)

features = ['Remain Pct','Leave Pct']



df = df.reindex()



df.plot(x='Area',y=features, sort_columns=True, kind='bar')
feature = ['Remain','Leave']

df = ref_results.sort_values(by=feature, ascending=True).head(20)

df.plot(x='Area',y=feature, kind='bar')
def decision(row):

    if row['Remain'] > row ['Leave']:

        return 'Remain'

    else:

        return 'Leave'



df = ref_results



df['Decision'] = df.apply(decision, axis=1)



decision_grp = df.groupby(by='Decision').count()

decision_grp.values[0][:0]
# Build correlation matrix

data = df[['Electorate','Percent Turnout','Votes Cast','Remain Pct']]



#axes = pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

#corr = data.corr().as_matrix()

#for i, j in zip(*np.triu_indices_from(axes, k=1)):

#    axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')

    

log_data = np.log(data)



# Produce a scatter matrix for each pair of newly-transformed features

pd.scatter_matrix(log_data, alpha = 0.3, figsize = (8,6), diagonal = 'kde');
ref_census = pd.read_csv('../input/census.csv')

ref_census.describe()

ref_census.columns

ref_census.head(5)