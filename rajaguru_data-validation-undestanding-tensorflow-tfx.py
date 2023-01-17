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
!pip install --user tensorflow_data_validation
import tensorflow_data_validation as tfdv
dipole_moments=pd.read_csv("../input/dipole_moments.csv")

magnetic_shielding_tensors=pd.read_csv("../input/magnetic_shielding_tensors.csv")

mulliken_charges=pd.read_csv("../input/mulliken_charges.csv")

potential_energy=pd.read_csv("../input/potential_energy.csv")

structures=pd.read_csv("../input/structures.csv")

scalar_coupling_contributions=pd.read_csv("../input/scalar_coupling_contributions.csv")

#dipole_moments=pd.read_csv("../input/dipole_moments.csv")

#dipole_moments=tfdv.generate_statistics_from_csv('../input/dipole_moments.csv')

#magnetic_shielding_tensors=tfdv.generate_statistics_from_csv('../input/magnetic_shielding_tensors.csv')

#mulliken_charges=tfdv.generate_statistics_from_csv('../input/mulliken_charges.csv')

#potential_energy=tfdv.generate_statistics_from_csv('../input/potential_energy.csv')

#scalar_coupling_contributions=tfdv.generate_statistics_from_csv('../input/scalar_coupling_contributions.csv')
dipole_moments_stats=tfdv.generate_statistics_from_dataframe(dipole_moments)

tfdv.visualize_statistics(dipole_moments_stats)
magnetic_shielding_tensors_stats=tfdv.generate_statistics_from_dataframe(magnetic_shielding_tensors)

tfdv.visualize_statistics(magnetic_shielding_tensors_stats)
mulliken_charges_stats=tfdv.generate_statistics_from_dataframe(mulliken_charges)

tfdv.visualize_statistics(mulliken_charges_stats)
potential_energy_stats=tfdv.generate_statistics_from_dataframe(potential_energy)

tfdv.visualize_statistics(potential_energy_stats)
structures_stats=tfdv.generate_statistics_from_dataframe(structures)

tfdv.visualize_statistics(structures_stats)
scalar_coupling_contributions_stats=tfdv.generate_statistics_from_dataframe(scalar_coupling_contributions)

tfdv.visualize_statistics(scalar_coupling_contributions_stats)
dipole_moments_schema=tfdv.infer_schema(dipole_moments_stats)

tfdv.write_schema_text(dipole_moments_schema,"dipole_moments_schema")



magnetic_shielding_tensors_schema=tfdv.infer_schema(magnetic_shielding_tensors_stats)

tfdv.write_schema_text(magnetic_shielding_tensors_schema,"magnetic_shielding_tensors_schema")



mulliken_charges_schema=tfdv.infer_schema(mulliken_charges_stats)

tfdv.write_schema_text(mulliken_charges_schema,"mulliken_charges_schema")



potential_energy_schema=tfdv.infer_schema(potential_energy_stats)

tfdv.write_schema_text(potential_energy_schema,"potential_energy_schema")



structures_schema=tfdv.infer_schema(structures_stats)

tfdv.write_schema_text(structures_schema,"structures_schema")



scalar_coupling_contributions_schema=tfdv.infer_schema(scalar_coupling_contributions_stats)

tfdv.write_schema_text(scalar_coupling_contributions_schema,"scalar_coupling_contributions_schema")

print(os.listdir(".")),tfdv.load_schema_text('magnetic_shielding_tensors_schema')
tfdv.display_schema(dipole_moments_schema)
tfdv.display_schema(magnetic_shielding_tensors_schema)
tfdv.display_schema(mulliken_charges_schema)
tfdv.display_schema(potential_energy_schema)
tfdv.display_schema(scalar_coupling_contributions_schema)
tfdv.display_schema(structures_schema)
train=pd.read_csv('../input/train.csv')

train.head()
test=pd.read_csv('../input/test.csv')

test.head()
def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)

train.shape,test.shape
train.head()
test.head()
train_stats=tfdv.generate_statistics_from_dataframe(train)
test_stats=tfdv.generate_statistics_from_dataframe(test)
train_schema=tfdv.infer_schema(train_stats)

tfdv.write_schema_text(train_schema,'train_schema')
test_schema=tfdv.infer_schema(test_stats)

tfdv.write_schema_text(test_schema,'test_schema')
anomalies=tfdv.validate_statistics(test_stats,train_schema)

tfdv.display_anomalies(anomalies)
tfdv.visualize_statistics(lhs_statistics=test_stats, rhs_statistics=train_stats,

                          lhs_name='TEST_DATASET', rhs_name='TRAIN_DATASET')
