# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
structures = pd.read_csv('../input/structures.csv')



structures
xyz = structures[['x','y','z']].values

ss = structures.groupby('molecule_name').size()

ss = ss.cumsum()

ss
ssx = np.zeros(len(ss) + 1, 'int')

ssx[1:] = ss

ssx
molecule_id = 20

print(ss.index[molecule_id])

start_molecule = ssx[molecule_id]

end_molecule = ssx[molecule_id+1]

xyz[start_molecule:end_molecule]
structures_idx = structures.set_index('molecule_name')
structures_idx.loc['dsgdb9nsd_000022'][['x', 'y', 'z']].values
def get_fast_dist_matrix(xyz, ssx, molecule_id):

    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]

    locs = xyz[start_molecule:end_molecule]    

    num_atoms = end_molecule - start_molecule

    loc_tile = np.tile(locs.T, (num_atoms,1,1))

    dist_mat = np.sqrt((loc_tile - loc_tile.T)**2).sum(axis=1)

    return dist_mat
molecule_id = 0

molecule = ss.index[molecule_id]

print(molecule)

get_fast_dist_matrix(xyz, ssx, molecule_id)
print(ss.index[molecule_id])
structures.loc[structures['molecule_name']=='dsgdb9nsd_000001']
carbon_x = -0.012698

carbon_y = 1.085804

carbon_z = 0.008001
def distance_from_carbon_to_xyz(xyz):

    return np.sqrt( (carbon_x - xyz[0])**2 + (carbon_y - xyz[1])**2 + (carbon_z - xyz[2])**2)
structures.loc[structures['molecule_name']=='dsgdb9nsd_000001', ['x','y','z']].apply(distance_from_carbon_to_xyz, 1)
molecule_id = 0

molecule = ss.index[molecule_id]

print(molecule)

get_fast_dist_matrix(xyz, ssx, molecule_id)