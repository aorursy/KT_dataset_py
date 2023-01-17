import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')
voters = pd.read_csv('/kaggle/input/philippine-voters-profile/2016_voters_profile.csv')

voters.head()
voters['literacy'] = voters['literacy'].str.replace('%','').astype(float)



sex_cols = ['male', 'female']

age_cols = ['17-19', '20-24', '25-29','30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64','65-above']

civil_status_cols = ['single', 'married', 'widow', 'legally_seperated']
per_region = voters.groupby('region').sum()

nrow=3

ncol=6

fig, axes = plt.subplots(nrow, ncol,figsize=(20,10),sharex=True,sharey=True)



reions = per_region.index

count=0

for r in range(nrow):

    for c in range(ncol):

        if(count==len(reions)):

            break

        col = reions[count]

        per_region.loc[col,age_cols].plot(kind='bar',ax=axes[r,c])

        axes[r,c].set_title(col)

        count = count+1
nrow=3

ncol=6

fig, axes = plt.subplots(nrow, ncol,figsize=(20,10),sharex=True,sharey=True)



reions = per_region.index

count=0

for r in range(nrow):

    for c in range(ncol):

        if(count==len(reions)):

            break

        col = reions[count]

        per_region.loc[col,sex_cols].plot(kind='bar',ax=axes[r,c],color=['blue','pink'])

        axes[r,c].set_title(col)

        count = count+1
nrow=3

ncol=6

fig, axes = plt.subplots(nrow, ncol,figsize=(20,10),sharex=True,sharey=True)



reions = per_region.index

count=0

for r in range(nrow):

    for c in range(ncol):

        if(count==len(reions)):

            break

        col = reions[count]

        per_region.loc[col,civil_status_cols].plot(kind='bar',ax=axes[r,c])

        axes[r,c].set_title(col)

        count = count+1