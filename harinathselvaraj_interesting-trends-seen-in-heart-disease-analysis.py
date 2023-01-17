import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

import altair as alt

alt.renderers.enable('notebook')



import os

print(os.listdir("../input"))
#Import dataset and print the structure of the dataset

df = pd.read_csv('../input/heart.csv')

print('\nShape of Dataset: {}'.format(df.shape))
#view the data

df.head()
#Checking the count of samples across each person's Age

df["age"].value_counts().head(10)
# Adding a new column - 'age_group'. Calculation logic - eg) If age is between 40 to 49, age_group will be 40



# start stop and step variables 

start, stop, step = 0, 1, 1  

# converting to string data type 

df["age_str"]= df["age"].astype(str) 

# slicing till 2nd last element 

df["age_group"]= df["age_str"].str.slice(start, stop, step) 

# concatenate zero at the end

df['age_group'] = df['age_group'] + '0'

#converting to int

df['age_group'] = df['age_group'].astype(int)
df2 = df.groupby(['age_group','target'])['age_group'].count().unstack('target').fillna(0)

df2.plot(kind='bar', stacked=True, color=['green', 'red'])
df = df.drop(columns=['age_group','age_str'])
#sex 1- male, 0-female

df[['age','sex','target']].groupby(['sex','target']).count()
#Chart showing the comparison heart diseases between Males & Females.

#sex 1- male, 0-female

df2 = df.groupby(['sex','target'])['sex'].count().unstack('target').fillna(0)

df2.plot(kind='bar', stacked=False, color=['limegreen', 'orangered'])

plt.show()
corr = df.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df.columns)

ax.set_yticklabels(df.columns)

plt.show()
# Grouping by 'cp' (Chest Pain Type)

df[['target','cp']].groupby(['cp']).count()
alt.Chart(df).mark_bar().encode(

    x='count(target):Q',

    y=alt.Y(

        'cp:N',

        sort=alt.EncodingSortField(

            field="target",  # The field to use for the sort

            op="count",  # The operation to run on the field prior to sorting

            order="descending"  # The order to sort in

        )

    )

)