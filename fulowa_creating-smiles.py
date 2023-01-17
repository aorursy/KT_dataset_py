import numpy as np # linear algebra

from scipy.stats.stats import pearsonr

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook as tqdm

import seaborn as sns 

import matplotlib.pyplot as plt

sns.set()

import os

import re
def load_dir_csv(directory, csv_files=None):

    if csv_files is None:

        csv_files = sorted( [ f for f in os.listdir(directory) if f.endswith(".csv") ])    

    csv_vars  = [ filename[:-4] for filename in csv_files ]

    gdict = globals()

    for filename, var in zip( csv_files, csv_vars ):

        print(f"{var:32s} = pd.read_csv({directory}/{filename})")

        gdict[var] = pd.read_csv( f"{directory}/{filename}" )

        print(f"{'nb of cols ':32s} = " + str(len(gdict[var])))

        display(gdict[var].head())



load_dir_csv("../input/", 

             ["train.csv", "structures.csv", "mulliken_charges.csv"])

!conda install -y -c openbabel openbabel 

import openbabel as ob
obConversion = ob.OBConversion()

obConversion.SetInFormat("xyz")



structdir='../input/structures/'

mols=[]

mols_files=os.listdir(structdir)

mols_index=dict(map(reversed,enumerate(mols_files)))

for f in mols_index.keys():

    mol = ob.OBMol()

    obConversion.ReadFile(mol, structdir+f) 

    mols.append(mol)
obConversion.SetOutFormat("smiles")
smiles = []

molecule_names = []
%%time

for mol_ in mols:

    smiles.append(re.split(r'\t+', obConversion.WriteString(mol_))[0])

    molecule_names.append(re.split(r'\t+', obConversion.WriteString(mol_))[1])
molecule_name_clean = []
for molecule_name in molecule_names: 

    molecule_name_clean.append(re.findall(r"[\w']+", molecule_name)[-2])
df_smiles = pd.DataFrame({'molecule_name': molecule_name_clean, 'smiles': smiles})
df_smiles.head()
df_smiles.shape
len(structures['molecule_name'].unique())