import numpy as np 

import pandas as pd 

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

import math

from numpy.linalg import svd

import ase

from ase import Atoms

import ase.visualize

import warnings

warnings.filterwarnings('ignore')



def view(molecule):

    # Select a molecule

    mol = struct[struct['molecule_name'] == molecule]

    

    # Get atomic coordinates

    xcart = mol.iloc[:, 3:].values

    

    # Get atomic symbols

    symbols = mol.iloc[:, 2].values

    

    # Display molecule

    system = Atoms(positions=xcart, symbols=symbols)

    print('Molecule Name: %s.' %molecule)

    return ase.visualize.view(system, viewer="x3d")
train = pd.read_csv('../input/train.csv', nrows=500_000)

struct = pd.read_csv('../input/structures.csv')
#quick PCA via singular value decomp - obtain principal eigenvector of molecule axis

def PCA_SVD(a):

    a_mean = a.mean(axis=1)

    #normalise

    a = (a.T - a_mean.T).T

    u, s, v = svd(a.T)

    return(v[0])



#obtain plane with minimum sum distance from nuclei

def get_plane(a):

    a = np.reshape(a, (a.shape[0], -1))

    centroid = a.mean(axis=1)

    #normalise

    x = a - centroid[:,np.newaxis]

    m = np.dot(x, x.T) 

    return(centroid, svd(m)[0][:,-1])
size_list = []

flatness = []

bond_angle_plane = []

bond_angle_axis = []



molecule_names = np.unique(train.molecule_name.values.flatten())



for i in tqdm(range(0, len(molecule_names))):

    

    mol = molecule_names[i]

    #obtain structure and bond information for each molecule

    temp_struct = struct.loc[struct.molecule_name==mol, :]

    bonds = train.loc[train.molecule_name == mol, :]

    

    #number of constituent atoms

    size = len(temp_struct)

    size_list.extend(np.full(len(bonds), size))

    

    #nuclei coords

    coords = np.column_stack([temp_struct.x.values,

                                     temp_struct.y.values,

                                     temp_struct.z.values]).T

    

    #principal axis of molecular alignment

    axis_vector = PCA_SVD(coords)

    

    #obtain flatness metric and plane angles for binds if nuclei > 3

    if size > 3:

        

        coords = coords - coords.mean()

        #calculate best fitting 2D plane

        ctr, norm = get_plane(coords) 

        

        #calculate distances of each atom from plane

        dists = np.zeros(size)

        for j in range(0, size):

            dists[j] = np.dot(norm, coords[:, j] - ctr)

                        

        #append mean distance to molecule list

        dist = abs(dists).sum()/len(dists)

        flatness.extend(np.full(len(bonds), dist))   

        

        #obtain angle subtended by each atomic bind and plane

        #print('There are {} bonds to calculate.'.format(len(bonds)))

        for j in range(0, len(bonds)):

            

            #obtain atom index numbers for this bond

            atom1 = bonds.atom_index_0.values[j]

            atom2 = bonds.atom_index_1.values[j]

            

            #get 3D coords

            atom1_coords = coords[:, atom1]

            atom2_coords = coords[:, atom2]

            

            #bond vector

            atom_vec = np.array([atom1_coords[0] - atom2_coords[0],

                                 atom1_coords[1] - atom2_coords[1],

                                 atom1_coords[2] - atom2_coords[2]])

            

            #angle subtended by bond vector and molecule plane

            angle = np.dot(norm, atom_vec)/(np.linalg.norm(norm)*np.linalg.norm(atom_vec))

            axis_angle = np.dot(axis_vector, atom_vec)/(np.linalg.norm(norm)*np.linalg.norm(atom_vec))

            

            #standardise to degrees <= 90

            angle = math.degrees(np.arccos(angle))

            axis_angle = math.degrees(np.arccos(axis_angle))



            if angle > 90:

                angle = 180 - angle

            

            if axis_angle > 90:

                axis_angle = 180 - axis_angle

                

            #not needed but somewhat easier to visualise

            angle = 90 - angle

            axis_angle = 90 - axis_angle

            bond_angle_plane.append(angle)

            bond_angle_axis.append(axis_angle)

               

    else:

        #for smaller molecules flatness and plane angle measurements are not informative

        flatness.extend(np.full(len(bonds), np.nan))

        bond_angle_plane.extend(np.full(len(bonds), np.nan))

        

        for j in range(0, len(bonds)):

            

            #obtain atom index numbers for this bond

            atom1 = bonds.atom_index_0.values[j]

            atom2 = bonds.atom_index_1.values[j]

            

            #get 3D coords

            atom1_coords = coords[:, atom1]

            atom2_coords = coords[:, atom2]

            

            #bond vector

            atom_vec = np.array([atom1_coords[0] - atom2_coords[0],

                                 atom1_coords[1] - atom2_coords[1],

                                 atom1_coords[2] - atom2_coords[2]])

            

            #angle subtended by bond vector and molecule principal axis

            axis_angle = np.dot(axis_vector, atom_vec)/(np.linalg.norm(axis_vector)*np.linalg.norm(atom_vec))

            

            #standardise to degrees <= 90

            axis_angle = math.degrees(np.arccos(axis_angle))

                 

            if axis_angle > 90:

                axis_angle = 180 - axis_angle

                

            #not needed but somewhat easier to visualise

            axis_angle = 90 - axis_angle

            bond_angle_axis.append(axis_angle)

  

    

train['num_atoms'] = np.asarray(size_list)

train['flatness_metric'] = np.asarray(flatness)

train['bond_angle_plane'] = np.asarray(bond_angle_plane)

train['bond_angle_axis'] = np.asarray(bond_angle_axis)
train.sort_values('flatness_metric', ascending=False).head(3)
view('dsgdb9nsd_003491')
train.sort_values('flatness_metric').head(3)
view('dsgdb9nsd_000050')
train.sort_values('bond_angle_plane', ascending=False).head(3)
view('dsgdb9nsd_000014')
train.sort_values('bond_angle_plane').head(3)
view('dsgdb9nsd_000023')
train.sort_values('bond_angle_axis', ascending=False).head(3)
view('dsgdb9nsd_000009')
train.sort_values('bond_angle_axis').head(3)
view('dsgdb9nsd_000861')
fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (18, 6))

train.loc[train.type=='2JHH', :].plot(kind='scatter', x='flatness_metric', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 0])

ax[0, 0].xaxis.set_label_text("")

ax[0, 0].set_title('2JHH')

train.loc[train.type=='1JHN', :].plot(kind='scatter', x='flatness_metric', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 1])

ax[0, 1].xaxis.set_label_text("")

ax[0, 1].set_title('1JHN')

train.loc[train.type=='3JHC', :].plot(kind='scatter', x='flatness_metric', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 0])

ax[1, 0].xaxis.set_label_text("")

ax[1, 0].set_title('3JHC')

train.loc[train.type=='2JHC', :].plot(kind='scatter', x='flatness_metric', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 1])

ax[1, 1].xaxis.set_label_text("")

ax[1, 1].set_title('2JHC')

fig.text(0.5, 0.04, 'FLATNESS METRIC', ha='center', va='center')

fig.text(0.06, 0.5, 'SCALAR COUPLING CONSTANT', ha='center', va='center', rotation='vertical')

plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (18, 6))

train.loc[train.type=='3JHH', :].plot(kind='scatter', x='flatness_metric', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 0])

ax[0, 0].xaxis.set_label_text("")

ax[0, 0].yaxis.set_label_text("")

ax[0, 0].set_title('3JHH')

train.loc[train.type=='1JHC', :].plot(kind='scatter', x='flatness_metric', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 1])

ax[0, 1].xaxis.set_label_text("")

ax[0, 1].yaxis.set_label_text("")

ax[0, 1].set_title('1JHC')

train.loc[train.type=='2JHN', :].plot(kind='scatter', x='flatness_metric', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 0])

ax[1, 0].xaxis.set_label_text("")

ax[1, 0].yaxis.set_label_text("")

ax[1, 0].set_title('2JHN')

train.loc[train.type=='3JHN', :].plot(kind='scatter', x='flatness_metric', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 1])

ax[1, 1].xaxis.set_label_text("")

ax[1, 1].yaxis.set_label_text("")

ax[1, 1].set_title('3JHN')

fig.text(0.5, 0.04, 'FLATNESS METRIC', ha='center', va='center')

fig.text(0.06, 0.5, 'SCALAR COUPLING CONSTANT', ha='center', va='center', rotation='vertical')

plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (18, 6))

train.loc[train.type=='2JHH', :].plot(kind='scatter', x='bond_angle_plane', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 0])

ax[0, 0].xaxis.set_label_text("")

ax[0, 0].yaxis.set_label_text("")

ax[0, 0].set_title('2JHH')

train.loc[train.type=='1JHN', :].plot(kind='scatter', x='bond_angle_plane', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 1])

ax[0, 1].xaxis.set_label_text("")

ax[0, 1].yaxis.set_label_text("")

ax[0, 1].set_title('1JHN')

train.loc[train.type=='3JHC', :].plot(kind='scatter', x='bond_angle_plane', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 0])

ax[1, 0].xaxis.set_label_text("")

ax[1, 0].yaxis.set_label_text("")

ax[1, 0].set_title('3JHC')

train.loc[train.type=='2JHC', :].plot(kind='scatter', x='bond_angle_plane', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 1])

ax[1, 1].xaxis.set_label_text("")

ax[1, 1].yaxis.set_label_text("")

ax[1, 1].set_title('2JHC')

fig.text(0.5, 0.04, 'BOND ANGLE - PLANE', ha='center', va='center')

fig.text(0.06, 0.5, 'SCALAR COUPLING CONSTANT', ha='center', va='center', rotation='vertical')

plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (18, 6))

train.loc[train.type=='3JHH', :].plot(kind='scatter', x='bond_angle_plane', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 0])

ax[0, 0].xaxis.set_label_text("")

ax[0, 0].yaxis.set_label_text("")

ax[0, 0].set_title('3JHH')

train.loc[train.type=='1JHC', :].plot(kind='scatter', x='bond_angle_plane', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 1])

ax[0, 1].xaxis.set_label_text("")

ax[0, 1].yaxis.set_label_text("")

ax[0, 1].set_title('1JHC')

train.loc[train.type=='2JHN', :].plot(kind='scatter', x='bond_angle_plane', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 0])

ax[1, 0].xaxis.set_label_text("")

ax[1, 0].yaxis.set_label_text("")

ax[1, 0].set_title('2JHN')

train.loc[train.type=='3JHN', :].plot(kind='scatter', x='bond_angle_plane', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 1])

ax[1, 1].xaxis.set_label_text("")

ax[1, 1].yaxis.set_label_text("")

ax[1, 1].set_title('3JHN')

fig.text(0.5, 0.04, 'BOND ANGLE - PLANE', ha='center', va='center')

fig.text(0.06, 0.5, 'SCALAR COUPLING CONSTANT', ha='center', va='center', rotation='vertical')

plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (18, 6))

train.loc[train.type=='2JHH', :].plot(kind='scatter', x='bond_angle_axis', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 0])

ax[0, 0].xaxis.set_label_text("")

ax[0, 0].yaxis.set_label_text("")

ax[0, 0].set_title('2JHH')

train.loc[train.type=='1JHN', :].plot(kind='scatter', x='bond_angle_axis', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 1])

ax[0, 1].xaxis.set_label_text("")

ax[0, 1].yaxis.set_label_text("")

ax[0, 1].set_title('1JHN')

train.loc[train.type=='3JHC', :].plot(kind='scatter', x='bond_angle_axis', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 0])

ax[1, 0].xaxis.set_label_text("")

ax[1, 0].yaxis.set_label_text("")

ax[1, 0].set_title('3JHC')

train.loc[train.type=='2JHC', :].plot(kind='scatter', x='bond_angle_axis', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 1])

ax[1, 1].xaxis.set_label_text("")

ax[1, 1].yaxis.set_label_text("")

ax[1, 1].set_title('2JHC')

fig.text(0.5, 0.04, 'BOND ANGLE - AXIS', ha='center', va='center')

fig.text(0.06, 0.5, 'SCALAR COUPLING CONSTANT', ha='center', va='center', rotation='vertical')

plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (18, 6))

train.loc[train.type=='3JHH', :].plot(kind='scatter', x='bond_angle_axis', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 0])

ax[0, 0].xaxis.set_label_text("")

ax[0, 0].yaxis.set_label_text("")

ax[0, 0].set_title('3JHH')

train.loc[train.type=='1JHC', :].plot(kind='scatter', x='bond_angle_axis', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[0, 1])

ax[0, 1].xaxis.set_label_text("")

ax[0, 1].yaxis.set_label_text("")

ax[0, 1].set_title('1JHC')

train.loc[train.type=='2JHN', :].plot(kind='scatter', x='bond_angle_axis', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 0])

ax[1, 0].xaxis.set_label_text("")

ax[1, 0].yaxis.set_label_text("")

ax[1, 0].set_title('2JHN')

train.loc[train.type=='3JHN', :].plot(kind='scatter', x='bond_angle_axis', y='scalar_coupling_constant', color='b', figsize=(15, 6), s=0.5, ax=ax[1, 1])

ax[1, 1].xaxis.set_label_text("")

ax[1, 1].yaxis.set_label_text("")

ax[1, 1].set_title('3JHN')

fig.text(0.5, 0.04, 'BOND ANGLE - AXIS', ha='center', va='center')

fig.text(0.06, 0.5, 'SCALAR COUPLING CONSTANT', ha='center', va='center', rotation='vertical')

plt.show()