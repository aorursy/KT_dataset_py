# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")

import missingno as msno
# import data

df = pd.read_csv('/kaggle/input/ufcdata/data.csv')

df.head()
df.tail()
df.describe()
df.describe(include='O')
# Check for Missing data

msno.matrix(df)
# Find the winning colour

fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.countplot(x="Winner", data=df, ax=ax)
# Find the common Referees

fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.countplot(x="Referee", data=df, ax=ax, order=df.Referee.value_counts().iloc[:10].index)

plt.xticks(rotation='vertical')
# Find Most common Number of Rounds

fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.countplot(x="no_of_rounds", data=df, ax=ax)
# How often a fight was a title bout

fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.countplot(x="title_bout", data=df, ax=ax)
# How often each weight class fought

fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.countplot(x="weight_class", data=df, ax=ax, order=df.weight_class.value_counts().iloc[:].index)

plt.xticks(rotation='vertical')
# weight class to title belt comparison

fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.countplot(x="weight_class", hue='title_bout', data=df, ax=ax, order=df.weight_class.value_counts().iloc[:].index)

plt.xticks(rotation='vertical')
# Most common Fighters

df_fighters = df.melt(value_vars=['R_fighter', 'B_fighter'], var_name='Color', value_name='Fighter')

fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.countplot(x="Fighter", data=df_fighters, ax=ax, order=df_fighters.Fighter.value_counts().iloc[:15].index)

plt.xticks(rotation='vertical')

# now see what sides were preferenced based on fighter

fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.countplot(x="Fighter", data=df_fighters, hue='Color', ax=ax, order=df_fighters.Fighter.value_counts().iloc[:15].index)

ax.legend(['Red Fighter', 'Blue Fighter'], facecolor='w')

plt.xticks(rotation='vertical')
# Fights by year

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df['Year'] = df['date'].dt.year

fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.countplot(x="Year", data=df, ax=ax)

plt.xticks(rotation='vertical')
# Most common Age of fighter

df_age = df.melt(value_vars=['R_age', 'B_age'], var_name='Color', value_name='Age')

fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.countplot(x="Age", data=df_age, ax=ax)

plt.xticks(rotation='vertical')