import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import sys

import os
df = pd.read_csv("../input/Admission_Predict.csv",sep = ",")

df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})

df=df.rename(columns = {'LOR ':'LOR'})

df.head(5)
print("There are",len(df), "data in this dataset.")

print(len(df.loc[df['Chance of Admit']>=0.70]), "of them have 'Chance of Admit' percentage higher than 70%.")
subjects = df.loc[df['Chance of Admit']>=0.70]

print("Highly Likely:",len(subjects.loc[subjects['Chance of Admit']>=0.9]))

print("Likely:",len(subjects.loc[(subjects['Chance of Admit']>=0.8) &(subjects['Chance of Admit']<0.9)]))

print("Reach:",len(subjects.loc[(subjects['Chance of Admit']>=0.7) &(subjects['Chance of Admit']<0.8)]))
# Setting three categories based on 'Chance of Admit' percentage

subjects.loc[subjects['Chance of Admit']>=0.9, 'Target'] = "Highly Likely"

subjects.loc[(subjects['Chance of Admit']>=0.8) &(subjects['Chance of Admit']<0.9), 'Target'] = "Likely"

subjects.loc[(subjects['Chance of Admit']>=0.7) &(subjects['Chance of Admit']<0.8), 'Target'] = "Reach"

subjects.Target=pd.Categorical(subjects.Target,categories=['Reach', 'Likely', 'Highly Likely'])

subjects=subjects.sort_values('Target')

subjects.head(5)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,10))

fig.subplots_adjust(hspace=0.3)

sns.boxplot(x="Target", y="GRE Score", data=subjects, palette="Set3", ax=axes[0,0])

sns.boxplot(x="Target", y="TOEFL Score", data=subjects, palette="Set3", ax=axes[0,1])

sns.boxplot(x="Target", y="SOP", data=subjects, palette="Set3", ax=axes[1,0])

sns.boxplot(x="Target", y="LOR", data=subjects, palette="Set3", ax=axes[1,1])
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16,14))

fig.subplots_adjust(hspace=0.3)

sns.violinplot(x="University Rating", y="GRE Score", data=subjects, palette="Set2", ax=axes[0,0])

sns.violinplot(x="University Rating", y="TOEFL Score", data=subjects, palette="Set2", ax=axes[0,1])

sns.violinplot(x="University Rating", y="SOP", data=subjects, palette="Set2", ax=axes[1,0])

sns.violinplot(x="University Rating", y="LOR", data=subjects, palette="Set2", ax=axes[1,1])

sns.violinplot(x="University Rating", y="CGPA", data=subjects, palette="Set2", ax=axes[2,0])

sns.violinplot(x="University Rating", y="Research", data=subjects, palette="Set2", ax=axes[2,1])
df['Target'] = 'Rest'

df.loc[df['Chance of Admit']>=0.9, 'Target'] = 'Chosen Ones'

df.head(5)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16,14))

fig.subplots_adjust(hspace=0.3)



sns.scatterplot(x="Chance of Admit", y="GRE Score", hue='Target', data=df, ax=axes[0,0])

sns.scatterplot(x="Chance of Admit", y="TOEFL Score", hue='Target',data=df, ax=axes[0,1])

sns.scatterplot(x="Chance of Admit", y="SOP", hue='Target',data=df, ax=axes[1,0])

sns.scatterplot(x="Chance of Admit", y="LOR", hue='Target',data=df, ax=axes[1,1])

sns.scatterplot(x="Chance of Admit", y="CGPA", hue='Target',data=df, ax=axes[2,0])

sns.scatterplot(x="Chance of Admit", y="Research", hue='Target', data=df, ax=axes[2,1])
df.corr().drop(['Serial No.'], axis=1)
sns.set(style="white")



# Compute the correlation matrix

corr = df.corr().drop(['Serial No.'], axis=1)



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(8, 8))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 210, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# These are students who meet the requirements of "The Chosen Ones" but still have lower 'Chance of Admit' percentage.

not_chosen = df.loc[(df["GRE Score"] >= 320)&(df["CGPA"] >= 9.0)&(df["Target"]=='Rest')]

print("However, there are", len(not_chosen), "people whose GRE Score and CGPA should have qualified as Highly Likely candidates.")

print("We will try to observe their characteristics compared to the Chosen Ones.")

chosen = df.loc[df['Target']=='Chosen Ones']

frames = [chosen, not_chosen]

result = pd.concat(frames)

result.corr().drop(['Serial No.'], axis=1)