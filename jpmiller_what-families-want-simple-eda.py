import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import colorcet as cc
fams = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv', 

                    index_col=['n_people','family_id'])

fams = fams.sort_values(list(fams), ascending=False)

fam_sizes = pd.Series(fams.index.get_level_values(0))



fams_array = fams.to_numpy()

pref_matrix = np.zeros((5000, 101), dtype=np.int8)

for i in range(5000):

    for j in range(10):

        day = fams_array[i, j]

        pref_matrix[i, day] = j+1



pref_matrix_exp = np.repeat(pref_matrix, fam_sizes, axis=0)

pref_matrix_exp.shape
sns.set()

plt.figure(figsize=(15,10))

heat_opts = {'cmap': ['#ffffff']+(cc.kbc),

             'xticklabels': 10,

             'yticklabels': False,

             'cbar_kws': {'label': 'Choice (top choice is dark)',

                          'shrink': 0.5

                          }

             }

ax=sns.heatmap(pref_matrix_exp, **heat_opts)

ax.invert_xaxis()

plt.xlabel('Days before Christmas', fontsize=14)

plt.ylabel('Families (larger families have taller blocks)',

           fontsize=14)

plt.title('Christmas Eve remains popular across a wide spectrum of preferred dates.', 

            fontsize=16, color='midnightblue')

plt.show()
fixed_costs = [500, 0, 50, 50, 100, 200, 200, 300, 300, 400, 500]

variable_costs = [434, 0, 0, 9, 9, 9, 18, 18, 36, 36, 235]



cost_matrix = np.zeros((5000, 101))

for i in range(5000):

    for j in range(101):

        choice = pref_matrix[i, j]

        cost = fixed_costs[choice] + variable_costs[choice]*fam_sizes[i]

        cost_matrix[i,j] = cost

        
fams_sorted = fam_sizes.sort_values(ascending=False).index

cost_matrix[fams_sorted]
sns.set()

plt.figure(figsize=(15,10))

heat_opts = {'cmap': cc.bmy,

             'xticklabels': 10,

             'yticklabels': False,

             'cbar_kws': {'label': 'Cost (USD)',

                          'shrink': 0.5

                          }

             }

ax=sns.heatmap(cost_matrix[fams_sorted], **heat_opts)

ax.invert_xaxis()

plt.xlabel('Days before Christmas', fontsize=14)

plt.ylabel('Families (larger families at top)', fontsize=14)

plt.title('Santa pays out if big families don\'t get their preference.', 

            fontsize=16, color='midnightblue')

plt.show()