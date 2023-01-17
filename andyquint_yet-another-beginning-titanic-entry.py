# Load libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Load data



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



both = pd.concat([train, test])
# Plot histograms for survivors vs nonsurvivors

survivors = train.loc[train['Survived'] == 1]

survivor_count = survivors.count()

nonsurvivors = train.loc[train['Survived'] == 0]

nonsurvivor_count = nonsurvivors.count()



fig, ax = plt.subplots(10,2, figsize=(12,20))

fig.tight_layout()

plt.subplots_adjust(left=0.4)



ax[0][0].set_title('Survivors')

ax[0][1].set_title('Nonsurvivors')



nbins = 20

x=(1,2)

width=0.4



# Male and female numbers

survivor_genders = (len(survivors.loc[survivors['Sex'] == 'male']), len(survivors.loc[survivors['Sex'] == 'female']))

nonsurvivor_genders = (len(nonsurvivors.loc[nonsurvivors['Sex'] == 'male']), len(nonsurvivors.loc[nonsurvivors['Sex'] == 'female']))



ax[0][0].bar(x,survivor_genders)

ax[0][1].bar(x,nonsurvivor_genders)

ax[0][0].set_xticks([i+width/2 for i in x])

ax[0][0].set_xticklabels(['Male','Female'])

ax[0][1].set_xticks([i+width/2 for i in x])

ax[0][1].set_xticklabels(['Male','Female'])

ax[0][0].set_ylabel('Sex', rotation=0, labelpad=80)



# Class numbers

survivor_class = (len(survivors.loc[survivors['Pclass'] == 1]), len(survivors.loc[survivors['Pclass'] == 2]), len(survivors.loc[survivors['Pclass'] == 3]))

nonsurvivor_class = (len(nonsurvivors.loc[nonsurvivors['Pclass'] == 1]), len(nonsurvivors.loc[nonsurvivors['Pclass'] == 2]), len(nonsurvivors.loc[nonsurvivors['Pclass'] == 3]))



x=(1,2,3)

ax[1][0].bar(x, survivor_class)

ax[1][1].bar(x, nonsurvivor_class)

ax[1][0].set_xticks([i+width/2 for i in x])

ax[1][0].set_xticklabels(['1','2','3'])

ax[1][1].set_xticks([i+width/2 for i in x])

ax[1][1].set_xticklabels(['1','2','3'])

ax[1][0].set_ylabel('Class', rotation=0, labelpad=80)



# Age histograms

ax[2][0].hist(survivors['Age'].dropna().tolist(), bins=nbins)

ax[2][1].hist(nonsurvivors['Age'].dropna().tolist(), bins=nbins)

ax[2][0].set_ylabel('Age', rotation=0, labelpad=80)



# Siblings/spouses

ax[3][0].hist(survivors['SibSp'].dropna().tolist(), bins=nbins)

ax[3][1].hist(nonsurvivors['SibSp'].dropna().tolist(), bins=nbins)

ax[3][0].set_ylabel('SibSp', rotation=0, labelpad=80)



# Parents/children

ax[4][0].hist(survivors['Parch'].dropna().tolist(), bins=nbins)

ax[4][1].hist(nonsurvivors['Parch'].dropna().tolist(), bins=nbins)

ax[4][0].set_ylabel('Parch', rotation=0, labelpad=80)



# Fare

ax[5][0].hist(survivors['Fare'].dropna().tolist(), bins=nbins)

ax[5][1].hist(nonsurvivors['Fare'].dropna().tolist(), bins=nbins)

ax[5][0].set_ylabel('Fare', rotation=0, labelpad=80)