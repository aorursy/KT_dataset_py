import os

import gc

import numpy as np

import pandas as pd

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

import math

pd.options.display.precision = 15
!ls ../input/
!conda install -y -c openbabel openbabel 

import openbabel

file_folder = '../input'

train = pd.read_csv(f'{file_folder}/train.csv')

test = pd.read_csv(f'{file_folder}/test.csv')

structures = pd.read_csv('../input/structures.csv')

x = structures.groupby('molecule_name').atom_index.max().reset_index(drop=False)

x.columns = ['molecule_name','totalatoms']

train = train.merge(x,on='molecule_name')

train = train[train.molecule_name=='dsgdb9nsd_000001']
obConversion = openbabel.OBConversion()

obConversion.SetInFormat("xyz")

structdir='../input/structures/'

mols=[]

mols_files=os.listdir(structdir)

mols_index=dict(map(reversed,enumerate(mols_files)))

for f in mols_index.keys():

    mol = openbabel.OBMol()

    obConversion.ReadFile(mol, structdir+f) 

    mols.append(mol)
stats = []

for m,groupdf in tqdm(train.groupby('molecule_name')):

    mol=mols[mols_index[m+'.xyz']]

    for i in groupdf.index.values:

        totalatoms = groupdf.loc[i].totalatoms

        firstatomid = int(groupdf.loc[i].atom_index_0)

        secondatomid = int(groupdf.loc[i].atom_index_1)

        entrystats = {}

        entrystats['totalatoms'] = totalatoms

        entrystats['scalar_coupling_constant'] = float(groupdf.loc[i].scalar_coupling_constant)

        entrystats['type'] = groupdf.loc[i]['type']

        a = mol.GetAtomById(firstatomid)

        b = mol.GetAtomById(secondatomid)

        entrystats['molecule_name'] = m

        entrystats['atom_index_0'] = firstatomid

        entrystats['atom_index_1'] = secondatomid

        entrystats['bond_distance'] = a.GetDistance(b)

        entrystats['bond_atom'] = b.GetType()

        entrystats['charge_H'] = a.GetPartialCharge()

        entrystats['charge_atom'] = b.GetPartialCharge()



        #Put the tertiary data in order of distance from first hydrogen

        tertiarystats = {}

        for j,c in enumerate(list(set(range(totalatoms)).difference(set([firstatomid,secondatomid])))):

            tertiaryatom = mol.GetAtomById(c)

            tp = tertiaryatom.GetType()

            dist = a.GetDistance(tertiaryatom)

            ang = a.GetAngle(b,tertiaryatom)*math.pi/180

            charge = tertiaryatom.GetPartialCharge()

            while(dist in tertiarystats):

                dist += 1e-15

                # print('Duplicates!',m,j,dist)

            tertiarystats[dist] = [tp,dist,ang,charge]

        

        for k, c in enumerate(sorted(tertiarystats.keys())):

            entrystats['tertiary_atom_'+str(k)] = tertiarystats[c][0]

            entrystats['tertiary_distance_'+str(k)] = tertiarystats[c][1]

            entrystats['tertiary_angle_'+str(k)] = tertiarystats[c][2]

            entrystats['tertiary_charge_'+str(k)] = tertiarystats[c][3]

        stats.append(entrystats)
obtrain = pd.DataFrame(stats)

obtrain.head(10)