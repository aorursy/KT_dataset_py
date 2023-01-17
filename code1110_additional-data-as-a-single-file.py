# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load data

# train_df = pd.read_csv('../input/train.csv')

potential_energy_df = pd.read_csv('../input/potential_energy.csv')

mulliken_charges_df = pd.read_csv('../input/mulliken_charges.csv')

scalar_coupling_contributions_df = pd.read_csv('../input/scalar_coupling_contributions.csv')

magnetic_shielding_tensors_df = pd.read_csv('../input/magnetic_shielding_tensors.csv')

dipole_moments_df = pd.read_csv('../input/dipole_moments.csv')

# structure_df = pd.read_csv('../input/structures.csv')

# test_df = pd.read_csv('../input/test.csv')



print("All the additional data were loaded.")
# combine "dipole_moments_df" and "potential_energy_df" (The both have 85003 rows)

DM_PE_df = pd.merge(dipole_moments_df, potential_energy_df, on='molecule_name')



# combine "magnetic_shielding_tensors_df" and "mulliken_charges_df" (The both have 1533537 rows)

MST_MC_df = pd.merge(magnetic_shielding_tensors_df, mulliken_charges_df, on=['molecule_name', 'atom_index'])



# combine these two

MST_MC_DM_PE_df = pd.merge(MST_MC_df, DM_PE_df, on='molecule_name', how='left')



# combine it with "scaler_coupling_contributions_df" 

combined_df = pd.merge(scalar_coupling_contributions_df, MST_MC_DM_PE_df, 

                           left_on=['molecule_name','atom_index_0'], right_on=['molecule_name','atom_index'], how='left')

combined_df1 = pd.merge(scalar_coupling_contributions_df, MST_MC_DM_PE_df, 

                           left_on=['molecule_name','atom_index_1'], right_on=['molecule_name','atom_index'], how='left')



# average values between atom_index_0 and atom_index_1

for c in list(MST_MC_DM_PE_df.columns.values)[2:]:

    combined_df[c] = (combined_df[c] + combined_df1[c]) / 2



combined_df = combined_df.drop(['atom_index'], axis=1)

print('Combined DF has {} rows and {} columns'.format(combined_df.shape[0], combined_df.shape[1]))



# save as csv

combined_df.to_csv('combined_additional_data.csv',index=False)



# head of the file

combined_df.head(12)