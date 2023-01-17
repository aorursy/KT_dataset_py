import numpy as np # linear algebra

import math

import h5py

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python.framework import ops

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import scatter_matrix

import os

print(os.listdir("../input"))
#Load data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#Load sub data

sub = pd.read_csv("../input/sample_submission.csv")

structures = pd.read_csv("../input/structures.csv")

scalar_coupling_contributions = pd.read_csv("../input/scalar_coupling_contributions.csv")

magnetic_shielding_tesors = pd.read_csv("../input/magnetic_shielding_tensors.csv")

dipole_moments = pd.read_csv("../input/dipole_moments.csv")

potential_energy = pd.read_csv("../input/potential_energy.csv")

mulliken_charges = pd.read_csv("../input/mulliken_charges.csv")
print(train.shape)

print(test.shape)

print(sub.shape)

print(structures.shape)

print(scalar_coupling_contributions.shape)

print(magnetic_shielding_tesors.shape)

print(dipole_moments.shape)

print(potential_energy.shape)

print(mulliken_charges.shape)



#check number of unique data 

print(train["molecule_name"].nunique())

print(test["molecule_name"].nunique())

print(structures["atom"].nunique())

print(train["type"].nunique())
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
train_structure = map_atom_info(train, 0)

train_structure = map_atom_info(train_structure, 1)
test_structure = map_atom_info(test, 0)

test_structure = map_atom_info(test_structure, 1)
train_p_0 = train_structure[['x_0', 'y_0', 'z_0']].values

train_p_1 = train_structure[['x_1', 'y_1', 'z_1']].values

train_structure['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test_p_0 = test_structure[['x_0', 'y_0', 'z_0']].values

test_p_1 = test_structure[['x_1', 'y_1', 'z_1']].values

test_structure['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
train_structure["atomic_number_0"]=train_structure["atom_0"].replace("H", 1).replace("C", 6).replace("N", 7).replace("O", 8).replace("F", 9)

train_structure["atomic_number_1"]=train_structure["atom_1"].replace("H", 1).replace("C", 6).replace("N", 7).replace("O", 8).replace("F", 9)
test_structure["atomic_number_0"]=test_structure["atom_0"].replace("H", 1).replace("C", 6).replace("N", 7).replace("O", 8).replace("F", 9)

test_structure["atomic_number_1"]=test_structure["atom_1"].replace("H", 1).replace("C", 6).replace("N", 7).replace("O", 8).replace("F", 9)
train_structure["coupling_number_0"]=train_structure["atom_0"].replace("H", 1).replace("C", 4).replace("N", 3).replace("O", 2).replace("F", 1)

train_structure["coupling_number_1"]=train_structure["atom_1"].replace("H", 1).replace("C", 4).replace("N", 3).replace("O", 2).replace("F", 1)
test_structure["coupling_number_0"]=test_structure["atom_0"].replace("H", 1).replace("C", 4).replace("N", 3).replace("O", 2).replace("F", 1)

test_structure["coupling_number_1"]=test_structure["atom_1"].replace("H", 1).replace("C", 4).replace("N", 3).replace("O", 2).replace("F", 1)
test_structure.head(10)
!pip install --quiet cupy-cuda100

!pip install --quiet chainer-chemistry

!pip install --quiet chaineripy

!conda install -y --quiet -c rdkit rdkit
# Check correctly installed, and modules can be imported.

import chainer

import chainer_chemistry

import chaineripy

import cupy

import rdkit



print('chainer version: ', chainer.__version__)

print('cupy version: ', cupy.__version__)

print('chainer-chemistry version: ', chainer_chemistry.__version__)

print('rdkit version:', rdkit.__version__)



from contextlib import contextmanager

import gc

from pathlib import Path

from time import time, perf_counter



import seaborn as sns

from rdkit import Chem

from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
def timer(name):

    t0 = perf_counter()

    yield

    t1 = perf_counter()

    print('[{}] done in {:.3f} s'.format(name, t1-t0))
"""

Copied from

https://github.com/jensengroup/xyz2mol/blob/master/xyz2mol.py



Modified `chiral_stereo_check` method for this task's purpose.

"""

##

# Written by Jan H. Jensen based on this paper Yeonjoon Kim and Woo Youn Kim

# "Universal Structure Conversion Method for Organic Molecules: From Atomic Connectivity

# to Three-Dimensional Geometry" Bull. Korean Chem. Soc. 2015, Vol. 36, 1769-1777 DOI: 10.1002/bkcs.10334

#

from rdkit.Chem import AllChem

import itertools

from rdkit.Chem import rdmolops

from collections import defaultdict

import copy

import networkx as nx  # uncomment if you don't want to use "quick"/install networkx



global __ATOM_LIST__

__ATOM_LIST__ = [x.strip() for x in ['h ', 'he', \

                                     'li', 'be', 'b ', 'c ', 'n ', 'o ', 'f ', 'ne', \

                                     'na', 'mg', 'al', 'si', 'p ', 's ', 'cl', 'ar', \

                                     'k ', 'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', \

                                     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', \

                                     'rb', 'sr', 'y ', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', \

                                     'cd', 'in', 'sn', 'sb', 'te', 'i ', 'xe', \

                                     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', \

                                     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w ', 're', 'os', 'ir', 'pt', \

                                     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', \

                                     'fr', 'ra', 'ac', 'th', 'pa', 'u ', 'np', 'pu']]





def get_atom(atom):

    global __ATOM_LIST__

    atom = atom.lower()

    return __ATOM_LIST__.index(atom) + 1





def getUA(maxValence_list, valence_list):

    UA = []

    DU = []

    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):

        if maxValence - valence > 0:

            UA.append(i)

            DU.append(maxValence - valence)

    return UA, DU





def get_BO(AC, UA, DU, valences, UA_pairs, quick):

    BO = AC.copy()

    DU_save = []



    while DU_save != DU:

        for i, j in UA_pairs:

            BO[i, j] += 1

            BO[j, i] += 1



        BO_valence = list(BO.sum(axis=1))

        DU_save = copy.copy(DU)

        UA, DU = getUA(valences, BO_valence)

        UA_pairs = get_UA_pairs(UA, AC, quick)[0]



    return BO





def valences_not_too_large(BO, valences):

    number_of_bonds_list = BO.sum(axis=1)

    for valence, number_of_bonds in zip(valences, number_of_bonds_list):

        if number_of_bonds > valence:

            return False



    return True





def BO_is_OK(BO, AC, charge, DU, atomic_valence_electrons, atomicNumList, charged_fragments):

    Q = 0  # total charge

    q_list = []

    if charged_fragments:

        BO_valences = list(BO.sum(axis=1))

        for i, atom in enumerate(atomicNumList):

            q = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

            Q += q

            if atom == 6:

                number_of_single_bonds_to_C = list(BO[i, :]).count(1)

                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:

                    Q += 1

                    q = 2

                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:

                    Q += 2

                    q = 1



            if q != 0:

                q_list.append(q)



    if (BO - AC).sum() == sum(DU) and charge == Q and len(q_list) <= abs(charge):

        return True

    else:

        return False





def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):

    if atom == 1:

        charge = 1 - BO_valence

    elif atom == 5:

        charge = 3 - BO_valence

    elif atom == 15 and BO_valence == 5:

        charge = 0

    elif atom == 16 and BO_valence == 6:

        charge = 0

    else:

        charge = atomic_valence_electrons - 8 + BO_valence



    return charge





def clean_charges(mol):

    # this hack should not be needed any more but is kept just in case

    #



    rxn_smarts = ['[N+:1]=[*:2]-[C-:3]>>[N+0:1]-[*:2]=[C-0:3]',

                  '[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',

                  '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]',

                  '[#8:1]=[#6:2]([!-:6])[*:3]=[*:4][#6-:5]>>[*-:1][*:2]([*:6])=[*:3][*:4]=[*+0:5]',

                  '[O:1]=[c:2][c-:3]>>[*-:1][*:2][*+0:3]',

                  '[O:1]=[C:2][C-:3]>>[*-:1][*:2]=[*+0:3]']



    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)



    for i, fragment in enumerate(fragments):

        for smarts in rxn_smarts:

            patt = Chem.MolFromSmarts(smarts.split(">>")[0])

            while fragment.HasSubstructMatch(patt):

                rxn = AllChem.ReactionFromSmarts(smarts)

                ps = rxn.RunReactants((fragment,))

                fragment = ps[0][0]

        if i == 0:

            mol = fragment

        else:

            mol = Chem.CombineMols(mol, fragment)



    return mol





def BO2mol(mol, BO_matrix, atomicNumList, atomic_valence_electrons, mol_charge, charged_fragments):

    # based on code written by Paolo Toscani



    l = len(BO_matrix)

    l2 = len(atomicNumList)

    BO_valences = list(BO_matrix.sum(axis=1))



    if (l != l2):

        raise RuntimeError('sizes of adjMat ({0:d}) and atomicNumList '

                           '{1:d} differ'.format(l, l2))



    rwMol = Chem.RWMol(mol)



    bondTypeDict = {

        1: Chem.BondType.SINGLE,

        2: Chem.BondType.DOUBLE,

        3: Chem.BondType.TRIPLE

    }



    for i in range(l):

        for j in range(i + 1, l):

            bo = int(round(BO_matrix[i, j]))

            if (bo == 0):

                continue

            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)

            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()



    if charged_fragments:

        mol = set_atomic_charges(mol, atomicNumList, atomic_valence_electrons, BO_valences, BO_matrix, mol_charge)

    else:

        mol = set_atomic_radicals(mol, atomicNumList, atomic_valence_electrons, BO_valences)



    return mol





def set_atomic_charges(mol, atomicNumList, atomic_valence_electrons, BO_valences, BO_matrix, mol_charge):

    q = 0

    for i, atom in enumerate(atomicNumList):

        a = mol.GetAtomWithIdx(i)

        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

        q += charge

        if atom == 6:

            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)

            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:

                q += 1

                charge = 0

            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:

                q += 2

                charge = 1



        if (abs(charge) > 0):

            a.SetFormalCharge(int(charge))



    # shouldn't be needed anymore bit is kept just in case

    # mol = clean_charges(mol)



    return mol





def set_atomic_radicals(mol, atomicNumList, atomic_valence_electrons, BO_valences):

    # The number of radical electrons = absolute atomic charge

    for i, atom in enumerate(atomicNumList):

        a = mol.GetAtomWithIdx(i)

        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])



        if (abs(charge) > 0):

            a.SetNumRadicalElectrons(abs(int(charge)))



    return mol





def get_bonds(UA, AC):

    bonds = []



    for k, i in enumerate(UA):

        for j in UA[k + 1:]:

            if AC[i, j] == 1:

                bonds.append(tuple(sorted([i, j])))



    return bonds





def get_UA_pairs(UA, AC, quick):

    bonds = get_bonds(UA, AC)

    if len(bonds) == 0:

        return [()]



    if quick:

        G = nx.Graph()

        G.add_edges_from(bonds)

        UA_pairs = [list(nx.max_weight_matching(G))]

        return UA_pairs



    max_atoms_in_combo = 0

    UA_pairs = [()]

    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):

        flat_list = [item for sublist in combo for item in sublist]

        atoms_in_combo = len(set(flat_list))

        if atoms_in_combo > max_atoms_in_combo:

            max_atoms_in_combo = atoms_in_combo

            UA_pairs = [combo]

        #           if quick and max_atoms_in_combo == 2*int(len(UA)/2):

        #               return UA_pairs

        elif atoms_in_combo == max_atoms_in_combo:

            UA_pairs.append(combo)



    return UA_pairs





def AC2BO(AC, atomicNumList, charge, charged_fragments, quick):

    # TODO

    atomic_valence = defaultdict(list)

    atomic_valence[1] = [1]

    atomic_valence[6] = [4]

    atomic_valence[7] = [4, 3]

    atomic_valence[8] = [2, 1]

    atomic_valence[9] = [1]

    atomic_valence[14] = [4]

    atomic_valence[15] = [5, 4, 3]

    atomic_valence[16] = [6, 4, 2]

    atomic_valence[17] = [1]

    atomic_valence[32] = [4]

    atomic_valence[35] = [1]

    atomic_valence[53] = [1]



    atomic_valence_electrons = {}

    atomic_valence_electrons[1] = 1

    atomic_valence_electrons[6] = 4

    atomic_valence_electrons[7] = 5

    atomic_valence_electrons[8] = 6

    atomic_valence_electrons[9] = 7

    atomic_valence_electrons[14] = 4

    atomic_valence_electrons[15] = 5

    atomic_valence_electrons[16] = 6

    atomic_valence_electrons[17] = 7

    atomic_valence_electrons[32] = 4

    atomic_valence_electrons[35] = 7

    atomic_valence_electrons[53] = 7



    # make a list of valences, e.g. for CO: [[4],[2,1]]

    valences_list_of_lists = []

    for atomicNum in atomicNumList:

        valences_list_of_lists.append(atomic_valence[atomicNum])



    # convert [[4],[2,1]] to [[4,2],[4,1]]

    valences_list = list(itertools.product(*valences_list_of_lists))



    best_BO = AC.copy()



    # implemenation of algorithm shown in Figure 2

    # UA: unsaturated atoms

    # DU: degree of unsaturation (u matrix in Figure)

    # best_BO: Bcurr in Figure

    #



    for valences in valences_list:

        AC_valence = list(AC.sum(axis=1))

        UA, DU_from_AC = getUA(valences, AC_valence)



        if len(UA) == 0 and BO_is_OK(AC, AC, charge, DU_from_AC, atomic_valence_electrons, atomicNumList,

                                     charged_fragments):

            return AC, atomic_valence_electrons



        UA_pairs_list = get_UA_pairs(UA, AC, quick)

        for UA_pairs in UA_pairs_list:

            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, quick)

            if BO_is_OK(BO, AC, charge, DU_from_AC, atomic_valence_electrons, atomicNumList, charged_fragments):

                return BO, atomic_valence_electrons



            elif BO.sum() >= best_BO.sum() and valences_not_too_large(BO, valences):

                best_BO = BO.copy()



    return best_BO, atomic_valence_electrons





def AC2mol(mol, AC, atomicNumList, charge, charged_fragments, quick):

    # convert AC matrix to bond order (BO) matrix

    BO, atomic_valence_electrons = AC2BO(AC, atomicNumList, charge, charged_fragments, quick)



    # add BO connectivity and charge info to mol object

    mol = BO2mol(mol, BO, atomicNumList, atomic_valence_electrons, charge, charged_fragments)



    return mol





def get_proto_mol(atomicNumList):

    mol = Chem.MolFromSmarts("[#" + str(atomicNumList[0]) + "]")

    rwMol = Chem.RWMol(mol)

    for i in range(1, len(atomicNumList)):

        a = Chem.Atom(atomicNumList[i])

        rwMol.AddAtom(a)



    mol = rwMol.GetMol()



    return mol





def get_atomicNumList(atomic_symbols):

    atomicNumList = []

    for symbol in atomic_symbols:

        atomicNumList.append(get_atom(symbol))

    return atomicNumList





def read_xyz_file(filename):

    atomic_symbols = []

    xyz_coordinates = []



    with open(filename, "r") as file:

        for line_number, line in enumerate(file):

            if line_number == 0:

                num_atoms = int(line)

            elif line_number == 1:

                if "charge=" in line:

                    charge = int(line.split("=")[1])

                else:

                    charge = 0

            else:

                atomic_symbol, x, y, z = line.split()

                atomic_symbols.append(atomic_symbol)

                xyz_coordinates.append([float(x), float(y), float(z)])



    atomicNumList = get_atomicNumList(atomic_symbols)



    return atomicNumList, charge, xyz_coordinates





def xyz2AC(atomicNumList, xyz):

    import numpy as np

    mol = get_proto_mol(atomicNumList)



    conf = Chem.Conformer(mol.GetNumAtoms())

    for i in range(mol.GetNumAtoms()):

        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))

    mol.AddConformer(conf)



    dMat = Chem.Get3DDistanceMatrix(mol)

    pt = Chem.GetPeriodicTable()



    num_atoms = len(atomicNumList)

    AC = np.zeros((num_atoms, num_atoms)).astype(int)



    for i in range(num_atoms):

        a_i = mol.GetAtomWithIdx(i)

        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * 1.30

        for j in range(i + 1, num_atoms):

            a_j = mol.GetAtomWithIdx(j)

            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * 1.30

            if dMat[i, j] <= Rcov_i + Rcov_j:

                AC[i, j] = 1

                AC[j, i] = 1



    return AC, mol





def chiral_stereo_check(mol):

    # Chem.SanitizeMol(mol)

    num_error = Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES, catchErrors=True)

    if num_error != 0:

        print('error id', num_error)



    Chem.DetectBondStereochemistry(mol, -1)

    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)

    Chem.AssignAtomChiralTagsFromStructure(mol, -1)



    return mol





def xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick):

    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,

    # and mol object with no connectivity information

    AC, mol = xyz2AC(atomicNumList, xyz_coordinates)



    # Convert AC to bond order matrix and add connectivity and charge info to mol object

    new_mol = AC2mol(mol, AC, atomicNumList, charge, charged_fragments, quick)



    # Check for stereocenters and chiral centers

    new_mol = chiral_stereo_check(new_mol)



    return new_mol
def mol_from_xyz(filepath, to_canonical=False):

    charged_fragments = True  # alternatively radicals are made



    # quick is faster for large systems but requires networkx

    # if you don't want to install networkx set quick=False and

    # uncomment 'import networkx as nx' at the top of the file

    quick = True



    atomicNumList, charge, xyz_coordinates = read_xyz_file(filepath)

    # print('atomicNumList', atomicNumList, 'charge', charge)

    mol = xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick)



    # Canonical hack

    if to_canonical:

        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        mol = Chem.MolFromSmiles(smiles)

    return mol
# This script is referred from http://rdkit.blogspot.jp/2015/02/new-drawing-code.html

# and http://cheminformist.itmol.com/TEST/wp-content/uploads/2015/07/rdkit_moldraw2d_2.html

from __future__ import print_function

from rdkit import Chem

from rdkit.Chem.Draw import IPythonConsole

from IPython.display import SVG



from rdkit.Chem import rdDepictor

from rdkit.Chem.Draw import rdMolDraw2D

def moltosvg(mol,molSize=(450,150),kekulize=True):

    mc = Chem.Mol(mol.ToBinary())

    if kekulize:

        try:

            Chem.Kekulize(mc)

        except:

            mc = Chem.Mol(mol.ToBinary())

    if not mc.GetNumConformers():

        rdDepictor.Compute2DCoords(mc)

    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])

    drawer.DrawMolecule(mc)

    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()

    return svg



def render_svg(svg):

    # It seems that the svg renderer used doesn't quite hit the spec.

    # Here are some fixes to make it work in the notebook, although I think

    # the underlying issue needs to be resolved at the generation step

    return SVG(svg.replace('svg:',''))
train_mol_names = train['molecule_name'].unique()

test_mol_names = test['molecule_name'].unique()

mol_names = np.hstack((train_mol_names, test_mol_names))

print(mol_names)
#This for loop need a few minutes calculation

input_dir = Path('../input')

mol_data=[]

for mol_id in mol_names:

    filepath = input_dir/f'structures/{mol_id}.xyz'

    mol_data.append(mol_from_xyz(filepath, to_canonical=True))
mol =pd.DataFrame({"molecule_name":mol_names, "mol_data":mol_data})
smiles=[]

for i in range(mol.shape[0]):

    smiles.append(Chem.MolToSmiles(mol.mol_data[i], isomericSmiles=False))
mol =pd.DataFrame({"molecule_name":mol_names, "mol_data":mol_data, "smiles":smiles})

#It looks success to convert xyz files

mol.head()
from rdkit import rdBase, Chem, DataStructs

print(rdBase.rdkitVersion) # 2017.09.1

from rdkit.Avalon import pyAvalonTools

from rdkit.Chem import AllChem

from rdkit.Chem.Fingerprints import FingerprintMols

from rdkit.Chem.AtomPairs import Pairs, Torsions

from rdkit.Chem import MACCSkeys

from rdkit.Chem import Descriptors

from rdkit.ML.Descriptors import MoleculeDescriptors
fps1 = [ MACCSkeys.GenMACCSKeys(mol1).ToBitString() for mol1 in mol.mol_data]

fps2 = [ list(map(int,list(fps))) for fps in fps1]

fps3 = np.array(fps2)
fps3.shape
#change data type ndarray to dataframe

fingerprints = pd.DataFrame(fps3)

fingerprints.head()
#merge dataset

mol = mol.join(fingerprints)

mol.head()
def calculate_descriptors(mols, names=None, ipc_avg=False):

    if names is None:

        names = [d[0] for d in Descriptors._descList]

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)

    descs = [calc.CalcDescriptors(mol) for mol in mols]

    descs = pd.DataFrame(descs, columns=names)

    if 'Ipc' in names and ipc_avg:

        descs['Ipc'] = [Descriptors.Ipc(mol, avg=True) for mol in mols]      

    return descs
descriptors = calculate_descriptors(mol.mol_data)
#Merge dataset

mol = mol.join(descriptors)

mol.head()
mol.to_csv('mol_dataset.csv',index=False)
def main():

    mols = mol.mol_data



    fps = [MACCSkeys.GenMACCSKeys(mol1) for mol1 in mols]

    fpMtx = np.array([fp2arr(fp) for fp in fps])



    plotPCA(fpMtx)



# convert rdkit fingerprint to numpy array

def fp2arr(fp):

    arr = np.zeros((1,))

    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr





# plot each compound using PCA

def plotPCA(fpMtx):

    from sklearn.decomposition import PCA



    pca = PCA(n_components=2)

    res = pca.fit_transform(fpMtx)

    # extract each component

    pc = res.T



    cum_cr = sum(pca.explained_variance_ratio_)

    print("cumulative contribution ratio=%.2f" % cum_cr)



    plt.figure()

    plt.scatter(pc[0], pc[1], marker=".")

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.legend()

    plt.savefig("pca_plot.png")



if __name__ == "__main__":

    main()