!conda install -y -c openbabel openbabel 

import openbabel as ob
import numpy as np

import pandas as pd

import seaborn as sns

import openbabel as ob

obConversion = ob.OBConversion()
Xtrain = pd.read_csv('../input/train.csv')

structures = pd.read_csv('../input/structures.csv')
Xtrain['type_0'] = Xtrain['type'].apply(lambda x: x[0]).astype(int)

Xtrain['type_1'] = Xtrain['type'].apply(lambda x: x[2:3])

Xtrain['type_2'] = Xtrain['type'].apply(lambda x: x[3:4])
Xtrain.head()
sns.violinplot(x="type_2", y="atom_index_1",

               split=True, data=Xtrain)
sns.violinplot(x="type_1", y="atom_index_0",

               split=True, data=Xtrain)


def read_ob_molecule(molecule_name, datadir="../input/structures"):

    mol = ob.OBMol()

    path = f"{datadir}/{molecule_name}.xyz"

    if not obConversion.ReadFile(mol, path):

        raise FileNotFoundError(f"Could not read molecule {path}")

    return mol
#Xtrain = Xtrain.iloc[:100000] # for speedup

Xtrain = Xtrain.sample(100000) # for speedup

df_structures = pd.read_csv('../input/structures.csv')

structures_idx = df_structures.set_index(["molecule_name"])

molecule_names = Xtrain.molecule_name.unique()

ob_natoms = []

ob_heavyatoms = []

ob_molweight = []

ob_molecule_name = []

ob_atom_index = []

for i,molecule_name in enumerate(molecule_names):

    if i%10000 ==0:

        print("OB molecule %d"%(i))

        

    # fill data for output DF

    ms = structures_idx.loc[molecule_name].sort_index()

    natoms = len(ms)

    ob_molecule_name.extend([molecule_name] * natoms)

    ob_atom_index.extend(ms.atom_index.values)

    

    # calculate open babel charge for each method

    mol = read_ob_molecule(molecule_name)

    assert (mol.NumAtoms() == natoms)

    mw = mol.GetMolWt()

    ob_heavyatoms.extend([mol.NumHvyAtoms()]*natoms)

    ob_natoms.extend([natoms]*natoms)

    ob_molweight.extend([mw]*natoms)

    

ob_data = pd.DataFrame({'molecule_name':ob_molecule_name,'natoms':ob_natoms,'nheavyatoms':  ob_heavyatoms, 'mw':ob_molweight,'atom_index':ob_atom_index})

ob_data.head(10)

    
Xtrain_ob = pd.merge(Xtrain, ob_data, how='left',

                        left_on=['molecule_name', 'atom_index_0'],

                        right_on=['molecule_name', 'atom_index'],suffixes=('_at1', '_at2'))
Xtrain_ob.head()
sns.violinplot(x="atom_index_0", y="natoms",

               split=True, data=Xtrain_ob)
Xtrain_ob.head()
Xtrain_ob['atom_index_sum'] = Xtrain_ob['atom_index_0']+Xtrain_ob['atom_index_1']

Xtrain_ob['atom_index_diff'] = Xtrain_ob['atom_index_0']-Xtrain_ob['atom_index_1']

Xtrain_ob['atom_index_abs'] = np.fabs(Xtrain_ob['atom_index_0']-Xtrain_ob['atom_index_1'])
def map_atom_info(df, structures, atom_idx):

    df = pd.merge(df, structures, how='left',

                  left_on=['molecule_name', f'atom_index_{atom_idx}'],

                  right_on=['molecule_name', 'atom_index'])



    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df
Xtrain_ob = Xtrain_ob.drop('atom_index', axis=1)

Xtrain_ob.head()


Xtrain_ob = map_atom_info(Xtrain_ob, structures, 0)

Xtrain_ob = map_atom_info(Xtrain_ob, structures, 1)
Xtrain_ob.head()
train_p_0 = Xtrain_ob[['x_0', 'y_0', 'z_0']].values

train_p_1 = Xtrain_ob[['x_1', 'y_1', 'z_1']].values

Xtrain_ob['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
sns.violinplot(x="atom_index_sum", y="natoms",

               split=True, data=Xtrain_ob)
sns.violinplot(x="atom_index_sum", y="natoms",

               split=False, data=Xtrain_ob.loc[Xtrain_ob.type=='2JHH'])