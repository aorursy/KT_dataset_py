#Import the modules

import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns

import altair as alt

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/parking-citations.csv')
#Seeing some sample data

df.head()
#Checking NaN in all the columns of the dataframe

df.isna().any()
df['Issue Date'].unique()
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

violation_count  = df['Violation Description'].value_counts()

violation_count = violation_count[:10,]

plt.figure(figsize=(11,8))

sns.barplot(violation_count.values, violation_count.index, alpha=1,palette=("Paired"))

plt.title('Top 10 Violation Types that were reported in Parking Citation', fontsize=22)

plt.xlabel('Number of Occurrences', size="20")

plt.ylabel('Violation Types', size="20")

plt.show()
sns.set(style="whitegrid")

state_plate_count  = df['RP State Plate'].value_counts()

state_plate_count = state_plate_count[:5,]

plt.figure(figsize=(11,8))

sns.barplot(state_plate_count.index, state_plate_count.values, alpha=1,palette=("Paired"))

plt.title('Top 5 state Number plates were reported in Parking Citation', fontsize=22)

plt.xlabel('Top State Number Plates', size="20")

plt.ylabel('Number of Occurrences', size="20")

plt.show()
#Calculate the 'Issue Year' from 'Issue Date'

df['Issue Year'] = df['Issue Date'].str[:4]
df['Issue Year'].value_counts()