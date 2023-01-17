# Imports

import numpy as np 

import pandas as pd 

import json

from scipy.spatial.distance import pdist, squareform



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Read JSON data (only one file for example)

with open("../input/pubChem_p_00000001_00025000.json") as f:

    data = json.load(f)
periodic_table = {'H':[1, 1.0079],

                  'C':[6, 12.0107],

                  'N':[7, 14.0067],

                  'O':[8, 15.9994],

                  'S':[16, 32.065],

                  'F':[9, 18.9984],

                  'Si':[14, 28.0855],

                  'P':[15, 30.9738],

                  'Cl':[17, 35.453],

                  'Br':[35, 79.904],

                  'I': [53, 126.9045]}
# Maximum number of atoms and the number of molecules

natMax = 50

nMolecules = len(data)



# Initiate arrays to store data

data_CM = np.zeros((nMolecules, natMax*(natMax+1)//2), dtype=float)

data_ids = np.zeros(nMolecules, dtype=int)

data_multipoles = np.zeros((nMolecules,14), dtype=float)

data_mmff94 = np.zeros(nMolecules, dtype=float)



# Loop over molecules and save

ind = 0

for molecule in data:

    

    # Check size, do not store molecules which has more atoms than natMax

    natoms = len(molecule['atoms'])

    if natoms > natMax:

        continue

    

    # Read energy, shape multipoles and Id

    data_mmff94[ind] = molecule['En']

    data_multipoles[ind,:] = molecule['shapeM']

    data_ids[ind] = molecule['id']

    

    # Initiate full CM padded with zeroes

    full_CM = np.zeros((natMax, natMax))

    full_Z = np.zeros(natMax)

    

    # Atoms: types and positions

    pos = []

    Z = []

    for i,at in enumerate(molecule['atoms']):

        Z.append(periodic_table[at['type']][0])

        pos.append(at['xyz'])

    

    pos = np.array(pos, dtype = float)

    Z = np.array(Z, dtype = float)

    

    # Construct Coulomb Matrices

    tiny = 1e-20    # A small constant to avoid division by 0

    dm = pdist(pos) # Pairwise distances



    # Coulomb matrix 

    coulomb_matrix = np.outer(Z,Z) / (squareform(dm) + tiny)

    full_CM[0:natoms, 0:natoms] = coulomb_matrix 

    full_Z[0:natoms] = Z



    # Coulomb vector (upper triangular part)

    iu = np.triu_indices(natMax,k=1)  # No diagonal k=1

    coulomb_vector = full_CM[iu]



    # Sort elements by decreasing order

    shuffle = np.argsort(-coulomb_vector)

    coulomb_vector = coulomb_vector[shuffle] # Unroll into vrctor



    # Construct feature vector

    coulomb_matrix = squareform(coulomb_vector)

    assert np.trace(coulomb_matrix) == 0, "Wrong Coulomb Matrix!"



    # Add diagonal terms

    coulomb_matrix += 0.5*np.power(full_Z,2.4)*np.eye(natMax) # Add the diagonal terms

    iu = np.triu_indices(natMax)                              # Upper diagonal 

    feature_vector = coulomb_matrix[iu]                       # Unroll into vector

    assert feature_vector.shape[0] == natMax*(natMax+1)//2, "Wrong feature dimensions"



    # Save data 

    data_CM[ind] = feature_vector

    

    # Iterate

    ind +=1

# Now save as pandas frame

dat = np.column_stack((data_CM, data_multipoles)) # Stack CM and multipole features

df = pd.DataFrame(dat)



# Column names

numfeats = np.shape(dat)[1]

cols = [x for x in range(1,numfeats+1,1)]

col_names = list(map(lambda x: 'f'+str(x), cols))

df.columns = col_names



# Add Energy and id

df.insert(0, 'pubchem_id', data_ids)

df['En'] = data_mmff94



# Save

path = "molecules_" + "1" + "_" + "25000" + ".csv"

df.to_csv(path)
df.head()