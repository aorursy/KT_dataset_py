!conda install openbabel -c openbabel -y
import sys

!conda install --yes --prefix {sys.prefix} -c rdkit rdkit
!pip install git+https://github.com/samoturk/mol2vec
import numpy as np

import pandas as pd

import os

from tqdm import tqdm_notebook as tqdm



from rdkit import Chem

from mol2vec.features import mol2alt_sentence

import pybel



from gensim.models import Word2Vec
file_dir = '../input/champs-scalar-coupling/structures/'

mols_files=os.listdir(file_dir)

mols_index=dict(map(reversed,enumerate(mols_files)))

mol_name = list(mols_index.keys())
# FROM: https://www.kaggle.com/roccomeli/easy-xyz-to-smiles-conversion



def xyz_to_smiles(fname: str) -> str:

    

    mol = next(pybel.readfile("xyz", fname))



    smi = mol.write(format="smi")



    return smi.split()[0].strip()
smiles = [xyz_to_smiles(file_dir + i) for i in tqdm(mol_name)]
df_smiles = pd.DataFrame({'molecule_name': mol_name, 'smiles': smiles})

df_smiles.head(11)
sentence = mol2alt_sentence(Chem.MolFromSmiles(df_smiles.smiles[33]), 1)

print('SMILE:', df_smiles.smiles[33])

print(sentence)
model = Word2Vec.load('../input/mol2vec/model_300dim.pkl')
model.wv[sentence[0]]