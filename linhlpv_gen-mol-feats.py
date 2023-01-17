# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline 

import pickle

from scipy.spatial import *

import os 
data_path = '../input/champsscalarold/'

structures_path = data_path + 'structures.csv'
structures_df = pd.read_csv(structures_path)
structures_df.head()
structures_df.describe()
eigen_value_distances_matrix_dict = {}

from tqdm import *
eigen_columns = list([f'eigen_value_{x+1}' for x in range(29)])
eigen_value_coulomb_matrix_columns = list([f'eigen_value_coulomb_matrix_{x+1}' for x in range(29)])
eigen_columns
eigen_value_coulomb_matrix_columns
eigen_value = pd.DataFrame(structures_df['molecule_name'].nunique(), index=np.arange(structures_df['molecule_name'].nunique()), columns=eigen_columns + ['molecule_name'], dtype=float)

atom_Z_dict = {'C': 6, 'O': 8, 'H': 1, 'N': 7, 'F': 9}
eigen_values = []

eigen_values_coulomb_matrix = []

eigen_value_coulomb_matrix_dict = {}

with tqdm(total=len(structures_df['molecule_name'].unique())) as pbar:

    for i, (name, df) in enumerate(structures_df.groupby('molecule_name')):

        xyz = df[['x', 'y', 'z']]

        matrix = distance_matrix(xyz, xyz)

        inverse_matrix = np.where(matrix != 0, 1/ matrix, 0)

        atom_type = df['atom'].values

        atom_Z = np.array([atom_Z_dict[type_] for type_ in atom_type])

        Z_ij = np.repeat(atom_Z, len(atom_Z)).reshape(-1, len(atom_Z), order='F')

        for i in range(len(atom_Z)):

            Z_ij[i] = Z_ij[i] * atom_Z[i]

        coulomb_matrix = np.multiply(Z_ij, inverse_matrix) + 1/2 * (np.multiply((np.eye(len(atom_Z))), Z_ij) ** 2.4)

        v_coulomb_matrix, _ = np.linalg.eig(coulomb_matrix)

        v_coulomb_matrix_pad = np.zeros(29)

        v_coulomb_matrix_pad[:v_coulomb_matrix.shape[0]] = v_coulomb_matrix

        

        v,_ = np.linalg.eig(matrix)

        v_pad = np.zeros(29)

        v_pad[:v.shape[0]] = v

        eigen_value_distances_matrix_dict[name] = v_pad

        eigen_values.append(v_pad)

        eigen_values_coulomb_matrix.append(v_coulomb_matrix_pad)

        eigen_value_coulomb_matrix_dict[name] = v_coulomb_matrix_pad

        

        pbar.update()
eigen_value = pd.DataFrame(eigen_values, columns = eigen_columns, dtype=float)

eigen_value['molecule_name'] = structures_df['molecule_name'].unique()

eigen_value
eigen_values_coulomb_matrix = pd.DataFrame(eigen_values_coulomb_matrix, columns = eigen_value_coulomb_matrix_columns, dtype=float)

eigen_values_coulomb_matrix['molecule_name'] = structures_df['molecule_name'].unique()

eigen_values_coulomb_matrix
with open('eigen_value_coulomb_matrix.pkl', 'wb') as f:

    pickle.dump(eigen_value_coulomb_matrix_dict, f)
with open('eigen_value_distance_matrix.pkl', 'wb') as f:

    pickle.dump(eigen_value_distances_matrix_dict, f)
eigen_value.to_csv('eigen_values.csv')
eigen_values_coulomb_matrix.to_csv('eigen_values_coulomb_matrix.csv')
atom_weight = {'C' : 12, 'F' : 19, 'O': 16, 'H': 1, 'N': 14}
molecule_weight_avg ={}
with tqdm(total=len(structures_df['molecule_name'].unique())) as pbar:

    for name, df in structures_df.groupby('molecule_name'):

        atom_type = df['atom']

        n_atoms = len(atom_type)

        atom_weights = np.array(list(map(lambda x: atom_weight[x], atom_type)))

#         print(atom_weights)

        molecule_weight_avg[name] = atom_weights.mean()

#         print(molecule_weight_avg)

        pbar.update()
with open('molecule_avg_weight.pkl', 'wb') as f:

    pickle.dump(molecule_weight_avg, f)
# with tqdm(total=len(structures_df['molecule_name']))