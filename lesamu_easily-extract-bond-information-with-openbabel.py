!conda install -c openbabel openbabel -y
import openbabel

import os

import pandas as pd
path = '../input/structures/'



# Initialize the OpenBabel object that we will use later.

obConversion = openbabel.OBConversion()

obConversion.SetInFormat("xyz")



# Define containers

mol_index = [] # This will be our dataframe index

bond_atom_0 = []

bond_atom_1 = []

bond_order = []

bond_length = []
for f in os.scandir(path):

    # Initialize an OBMol object

    mol = openbabel.OBMol()

    read_ok = obConversion.ReadFile(mol, f.path)

    if not read_ok:

        # There was an error reading the file

        raise Exception(f'Could not read file {f.path}')

    

    mol_name = f.name[:-4] 

    mol_index.extend([mol_name] * mol.NumBonds()) # We need one entry per bond

    

    # Extract bond information

    mol_bonds = openbabel.OBMolBondIter(mol) # iterate over all the bonds in the molecule

    for bond in mol_bonds:

        bond_atom_0.append(bond.GetBeginAtomIdx() - 1) # Must be 0-indexed

        bond_atom_1.append(bond.GetEndAtomIdx() - 1)

        bond_length.append(bond.GetLength())

        bond_order.append(bond.GetBondOrder())
# Put everything into a dataframe

df = pd.DataFrame({'molecule_name': mol_index,

                   'atom_0': bond_atom_0,

                   'atom_1': bond_atom_1,

                   'order': bond_order,

                   'length': bond_length})

    

df = df.sort_values(['molecule_name', 'atom_0', 'atom_1']).reset_index(drop=True)    



print(df.head(10))
# My favorite way of storing variables

import shelve

with shelve.open('vars.shelf') as shelf:

    shelf['bonds'] = df
# To load the variable later, use:

with shelve.open('vars.shelf') as shelf:

    bonds = shelf['bonds']