# Import modules.

# Linear algebra.
import numpy as np 
# Data processing, CSV file I/O (e.g. pd.read_csv).
import pandas as pd
# Graphs and charts.
import matplotlib.pyplot as plt
# Makes matplotlib suck less... much less.
import seaborn as sns
# Why do I need this again?
%matplotlib inline

# Read in the data.
train = pd.read_csv('../input/train.csv').set_index('PassengerId')
# Copy the relevant training data and clean it up.
age = train[['Age', 'Survived']].copy()
age['Age'] = [-1 if pd.isnull(x) else x for x in age['Age']]
age['Age'] = [np.floor(x) for x in age['Age']]

# Pivot on the age and reindex.
age = pd.pivot_table(age, index=['Age'], values=['Survived']).fillna(0)
age = age.reindex(np.arange(-1,81)).fillna(0)

# Create a bar plot.
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
sns.barplot(x=age.index.values, y='Survived', data=age, ax=ax)
ax.set_xticks([0,1,11,21,31,41,51,61,71,81])
ax.set_xticklabels([-1,0,10,20,30,40,50,60,70,80])
fig