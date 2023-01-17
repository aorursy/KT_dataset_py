# Importing required libraries.

import pandas as pd

import numpy as np

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation 

%matplotlib inline   

sns.set(color_codes=True)





df = pd.read_csv("../input/corruption-in-india/Cases_registered_under_PCA_act_and_related_sections_IPC_2013.csv")

df.head(5)
Pending = df['Pending Investigation From Previous Year']

Total = df['Total Cases For Investigation']

States = df['STATE/UT']
plt.scatter(Pending , Total)
plt.hist(Pending)
sns.distplot(Pending)
plt.barh(States,Pending)
plt.boxplot(Pending)