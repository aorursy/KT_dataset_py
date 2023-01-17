%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

train.rename(columns={

    'number_of_total_atoms': 'natoms',

    'percent_atom_al': 'al',

    'percent_atom_ga': 'ga',

    'percent_atom_in': 'in',

    'lattice_vector_1_ang': 'a',

    'lattice_vector_2_ang': 'b',

    'lattice_vector_3_ang': 'c',

    'lattice_angle_alpha_degree': 'alpha',

    'lattice_angle_beta_degree': 'beta',

    'lattice_angle_gamma_degree': 'gamma',

    'formation_energy_ev_natom': 'E0',

    'bandgap_energy_ev': 'bandgap'

}, inplace=True)
train['spacegroup_gamma'] = train.apply(

    lambda row: '{:03.0f}-{:03d}'.format(row['spacegroup'], int((row['gamma']+5)/10)*10),

    axis=1)

train['a_over_b'] = train.apply(lambda row: row['a'] / row['b'], axis=1)

train['c_over_b'] = train.apply(lambda row: row['c'] / row['b'], axis=1)
for i, sgg in enumerate(sorted(train['spacegroup_gamma'].unique())):

    train[train['spacegroup_gamma'] == sgg].plot(

        figsize=(16,8), kind='scatter', x='a_over_b', y='c_over_b', title=sgg,

        ax=plt.subplot2grid((2, 4), (i//4, i%4))

    )

plt.tight_layout()

plt.show()