import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

from itertools import permutations

%matplotlib inline



MIN_OCCUPANCY = 125

MAX_OCCUPANCY = 300

df = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col=0)

CHOICES = [c for c in df.columns if c.startswith('choice')]
sizes = Counter(df.n_people)

print(sizes)

df.n_people.hist(bins=7, range=(1.5, 8.5))
fig, axes = plt.subplots(nrows=2, ncols=5, sharex='all', sharey='all', figsize=(20, 4))

for col, ax in zip(CHOICES, axes.flatten()):

    df[[col, 'n_people']].groupby(col).sum().plot(ax=ax)

    ax.plot((0, 100), (125, 125), c='r', ls=':')

    ax.plot((0, 100), (300, 300), c='r', ls=':')

    ax.set_title(col)
fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all', sharey='all', figsize=(20, 4))

for i, col in enumerate(CHOICES):

    df[[col, 'n_people']].groupby(col).sum().plot(ax=ax, label=i)

ax.plot((0, 100), (125, 125), c='r', ls=':')

ax.plot((0, 100), (300, 300), c='r', ls=':')

ax.legend()
# Check to see if there's an obvious correlation between family size and priority choice

plt.subplots(figsize=(20, 4))

plt.scatter(df.choice_0, df.n_people)
def accounting_penalty(nd0, nd1):

    return max(0, (

        ((nd0 - 125) / 400)

        * nd0

        * (0.5 + ((nd0 - nd1) / 50))

    ))
acc_p_df = pd.DataFrame([*permutations(range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1), 2)], columns=['nd0', 'nd1'])

acc_p_df['penalty'] = acc_p_df.apply(lambda row: accounting_penalty(row.nd0, row.nd1), axis=1)

ax = sns.heatmap(acc_p_df.pivot(index='nd1', columns='nd0'))

ax.invert_yaxis()

def consolation(df: pd.DataFrame, family: int, day: int):

    members = df.loc[family, 'n_people']

    try:

        choice = int(np.where(df.loc[family].values.flatten() == day)[0])

    except TypeError:

        choice = 'x'

    cost = {

        0: 0,

        1: 50,

        2: 50 + 9 * members,

        3: 100 + 9 * members,

        4: 200 + 9 * members,

        5: 200 + 18 * members,

        6: 300 + 18 * members,

        7: 300 + 36 * members,

        8: 400 + 36 * members,

        9: 500 + 135 * members,

        'x':  500 + 434 * members,

    }

    

    print(members, choice)

    return cost[choice]