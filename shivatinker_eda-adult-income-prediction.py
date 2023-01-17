# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv("../input/adult.csv")
df.head()
greaterThan50 = df[df['income'] == ">50K"]
lessThan50 = df[df['income'] == "<=50K"]

ax = sns.countplot(greaterThan50['age'])
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
plt.tight_layout()
ax = sns.countplot(lessThan50['age'])
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
plt.tight_layout()
ax = sns.countplot(greaterThan50['education'])
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
plt.tight_layout()
fullMid = ["bachelors", "prof-school", "assoc-acdm", "assoc-voc", "masters", "doctorate"]
fullMidDF = greaterThan50.apply(lambda x: x['education'].lower() in fullMid, axis = 1)
len(fullMidDF[fullMidDF == True].index) / len(fullMidDF.index)
maleDF = df[df['gender'] == 'Male']
married = ['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent']
marriedDF = maleDF[maleDF.apply(lambda x: x['marital-status'] in married, axis = 1)]
unmarriedDF = maleDF[maleDF.apply(lambda x: x['marital-status'] not in married, axis = 1)]
print(len(marriedDF[marriedDF['income'] == '>50K'].index) / len(marriedDF.index))
print(len(unmarriedDF[unmarriedDF['income'] == '>50K'].index) / len(unmarriedDF.index))
maxHours = df['hours-per-week'].max()
print(maxHours)
maxHoursDF = df[df['hours-per-week'] == maxHours]
len(maxHoursDF[maxHoursDF['income'] == '>50K'].index) / len(maxHoursDF.index)
df["norm-income"] = df.apply(lambda x: 1 if x['income'] == '>50K' else 0, axis = 1)
df.corr()
z = df.groupby('education')['hours-per-week'].mean()
ax = sns.barplot(z.index, z.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
0
z = df.groupby('native-country')['norm-income'].mean().sort_values()
ax = sns.barplot(z.index, z.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
0
z = df.groupby('occupation')['norm-income'].mean().sort_values()
ax = sns.barplot(z.index, z.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
0