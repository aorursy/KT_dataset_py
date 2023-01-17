import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
df1 = pd.read_csv("../input/memory-test-on-drugged-islanders-data/Islander_data.csv")

df1.head()
df1.dtypes
df1.isnull().sum()
df1.describe()
df1
df1['age'].value_counts()
fig = plt.figure(figsize=(12, 5))

ax = fig.add_axes([0,0,1,1])

sns.countplot(x='Drug', data=df1, ax=ax)

plt.tight_layout()

plt.title('Breakdown of Observations by Drug')

plt.show()
df1.groupby('Drug')['Diff'].mean()
df1['avg_ovr_diff'] = df1.groupby('Drug')['Diff'].transform('mean')

df1['avg_ovr_diff']
fig = plt.figure(figsize=(12, 5))

ax = fig.add_axes([0,0,1,1])

sns.barplot(x=df1['Drug'], y=df1['avg_ovr_diff'], ax=ax)

plt.tight_layout()

plt.title('Overall Avg Difference in Memory Score by Drug \n (For all Ages)')

plt.show()
df1['age_group'] = pd.cut(

    df1['age'],

    np.arange(start=df1['age'].min(), step=5, stop=df1['age'].max())

)

df1[['age', 'age_group']].head()
df1.head()
df1['avg_age_diff'] = df1.groupby('age_group')['Diff'].transform('mean')

df1[['age_group', 'avg_age_diff']].head()
fig = plt.figure(figsize=(12, 5))

ax = fig.add_axes([0,0,1,1])

sns.countplot(x='age_group', data=df1, ax=ax)

plt.title('Number of Observations by Age Group')

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(12, 5))

ax = fig.add_axes([0,0,1,1])

sns.barplot(x=df1['age_group'], y=df1['avg_age_diff'], ax=ax)

plt.title('Avg Change in Memory Score by Age Group \n (Includes Both Drugs and Placebo)')

plt.tight_layout()

plt.show()
alprazolam_df = df1.loc[df1['Drug'] == 'A']

alprazolam_df
triazolam_df = df1.loc[df1['Drug'] == 'T']

triazolam_df
sugar_df = df1.loc[df1['Drug'] == 'S']

sugar_df
alprazolam_df['avg_age_diff'] = alprazolam_df.groupby('age_group')['Diff'].transform('mean')

alprazolam_df[['age_group', 'avg_age_diff']].head()
fig = plt.figure(figsize=(12, 5))

ax = fig.add_axes([0,0,1,1])

sns.barplot(x=alprazolam_df['age_group'], y=alprazolam_df['avg_age_diff'], ax=ax)

plt.title('Average Memory Score Difference \n (Alprazolam)')

plt.tight_layout()

plt.show()
triazolam_df['avg_age_diff'] = triazolam_df.groupby('age_group')['Diff'].transform('mean')

triazolam_df[['age_group', 'avg_age_diff']].head()
fig = plt.figure(figsize=(12, 5))

ax = fig.add_axes([0,0,1,1])

sns.barplot(x=triazolam_df['age_group'], y=triazolam_df['avg_age_diff'], ax=ax)

plt.plot('Average Memory Score Difference \n (Triazolam)')

plt.tight_layout()

plt.show()
sugar_df['avg_age_diff'] = sugar_df.groupby('age_group')['Diff'].transform('mean')

sugar_df[['age_group', 'avg_age_diff']].head()
fig = plt.figure(figsize=(12, 5))

ax = fig.add_axes([0,0,1,1])

sns.barplot(x=sugar_df['age_group'], y=sugar_df['avg_age_diff'], ax=ax)

plt.title('Average Memory Score Difference \n (Placebo)')

plt.tight_layout()

plt.show()