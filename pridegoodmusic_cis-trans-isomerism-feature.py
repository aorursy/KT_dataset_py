!conda install -c openbabel openbabel -y
import openbabel

import numpy as np

import pandas as pd

import os

from tqdm import tqdm

tqdm.pandas()
print(os.listdir('../input'))
base_path = '../input/champs-scalar-coupling'



structures = pd.read_csv(f'{base_path}/structures.csv')



# data reduction

train = pd.read_csv(f'{base_path}/train.csv')[::10]

test = pd.read_csv(f'{base_path}/test.csv')[::10]
obConversion = openbabel.OBConversion()

obConversion.SetInFormat("xyz")

xyz_path = f'{base_path}/structures/'
def cis_trans_bond_indices(molecule_name):

    mol = openbabel.OBMol()

    obConversion.ReadFile(mol, f'{xyz_path}/{molecule_name}.xyz')

    obs = openbabel.OBStereoFacade(mol)

    has_ct = [obs.HasCisTransStereo(n) for n in range(mol.NumBonds())]

    return [i for i, x in enumerate(has_ct) if x == True] if has_ct else []
df = pd.DataFrame(structures.molecule_name.unique(), columns=['molecule_name'])

df.head()
df['bond_indices'] = df.molecule_name.progress_apply(lambda x: cis_trans_bond_indices(x))

df['len_bond_indices'] = df.bond_indices.progress_apply(lambda x:len(x))
df.len_bond_indices.unique()
df[df['len_bond_indices']!=0].head()
train = pd.merge(train, df, how='left', on='molecule_name')

test = pd.merge(test, df, how='left', on='molecule_name')
def is_cis_trans(molecule_name, bond_indices, atom_index_0, atom_index_1):

    if len(bond_indices) == 0:

        return pd.Series([0,0])



    mol = openbabel.OBMol()

    obConversion.ReadFile(mol, f'{xyz_path}/{molecule_name}.xyz')

    obs = openbabel.OBStereoFacade(mol)

    

    is_cis   = [obs.GetCisTransStereo(i).IsCis(atom_index_0, atom_index_1) for i in bond_indices]

    is_trans = [obs.GetCisTransStereo(i).IsTrans(atom_index_0, atom_index_1) for i in bond_indices]

    return pd.Series([int(True in is_cis), int(True in is_trans)])
train[['is_cis','is_trans']] = train.progress_apply(lambda x: is_cis_trans(x.molecule_name,

                                                                           x.bond_indices,

                                                                           x.atom_index_0,

                                                                           x.atom_index_1), axis=1)
#test[['is_cis','is_trans']] = test.progress_apply(lambda x: is_Cis_Trans(x.molecule_name,

#                                                                         x.bond_indices,

#                                                                         x.atom_index_0,

#                                                                         x.atom_index_1), axis=1)
#train.to_csv('train_cis_trans.csv')

#test.to_csv('test_cis_trans.csv')
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from sympy.geometry import Point3D





init_notebook_mode(connected=True)



def plot_molecule(molecule_name, structures_df):

    """Creates a 3D plot of the molecule"""

    

    atomic_radii = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)  

    cpk_colors = dict(C='black', F='green', H='white', N='blue', O='red')



    molecule = structures_df[structures_df.molecule_name == molecule_name]

    coordinates = molecule[['x', 'y', 'z']].values

    x_coordinates = coordinates[:, 0]

    y_coordinates = coordinates[:, 1]

    z_coordinates = coordinates[:, 2]

    elements = molecule.atom.tolist()

    radii = [atomic_radii[element] for element in elements]

    

    def get_bonds():

        """Generates a set of bonds from atomic cartesian coordinates"""

        ids = np.arange(coordinates.shape[0])

        bonds = dict()

        coordinates_compare, radii_compare, ids_compare = coordinates, radii, ids

        

        for _ in range(len(ids)):

            coordinates_compare = np.roll(coordinates_compare, -1, axis=0)

            radii_compare = np.roll(radii_compare, -1, axis=0)

            ids_compare = np.roll(ids_compare, -1, axis=0)

            distances = np.linalg.norm(coordinates - coordinates_compare, axis=1)

            bond_distances = (radii + radii_compare) * 1.3

            mask = np.logical_and(distances > 0.1, distances <  bond_distances)

            distances = distances.round(2)

            new_bonds = {frozenset([i, j]): dist for i, j, dist in zip(ids[mask], ids_compare[mask], distances[mask])}

            bonds.update(new_bonds)

        return bonds            

            

    def atom_trace():

        """Creates an atom trace for the plot"""

        colors = [cpk_colors[element] for element in elements]

        markers = dict(color=colors, line=dict(color='lightgray', width=2), size=7, symbol='circle', opacity=0.8)

        trace = go.Scatter3d(x=x_coordinates, y=y_coordinates, z=z_coordinates, mode='markers', marker=markers,

                             text=elements, name='')

        return trace



    def bond_trace():

        """"Creates a bond trace for the plot"""

        trace = go.Scatter3d(x=[], y=[], z=[], hoverinfo='none', mode='lines',

                             marker=dict(color='grey', size=7, opacity=1))

        for i, j in bonds.keys():

            trace['x'] += (x_coordinates[i], x_coordinates[j], None)

            trace['y'] += (y_coordinates[i], y_coordinates[j], None)

            trace['z'] += (z_coordinates[i], z_coordinates[j], None)

        return trace

    

    bonds = get_bonds()

    

    zipped = zip(range(len(elements)), x_coordinates, y_coordinates, z_coordinates)

    annotations_id = [dict(text=num, x=x, y=y, z=z, showarrow=False, yshift=15)

                   for num, x, y, z in zipped]

    

    annotations_length = []

    for (i, j), dist in bonds.items():

        p_i, p_j = Point3D(coordinates[i]), Point3D(coordinates[j])

        p = p_i.midpoint(p_j)

        annotation = dict(text=dist, x=float(p.x), y=float(p.y), z=float(p.z), showarrow=False, yshift=15)

        annotations_length.append(annotation)   

    

    updatemenus = list([

        dict(buttons=list([

                 dict(label = 'Atom indices',

                      method = 'relayout',

                      args = [{'scene.annotations': annotations_id}]),

                 dict(label = 'Bond lengths',

                      method = 'relayout',

                      args = [{'scene.annotations': annotations_length}]),

                 dict(label = 'Atom indices & Bond lengths',

                      method = 'relayout',

                      args = [{'scene.annotations': annotations_id + annotations_length}]),

                 dict(label = 'Hide all',

                      method = 'relayout',

                      args = [{'scene.annotations': []}])

                 ]),

                 direction='down',

                 xanchor = 'left',

                 yanchor = 'top'

            ),        

    ])

    

    data = [atom_trace(), bond_trace()]

    axis_params = dict(showgrid=False, showticklabels=False, zeroline=False, titlefont=dict(color='white'),

                       showbackground=False)

    layout = dict(scene=dict(xaxis=axis_params, yaxis=axis_params, zaxis=axis_params, annotations=annotations_id), 

                  margin=dict(r=0, l=0, b=0, t=0), showlegend=False, updatemenus=updatemenus)



    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
cis_mol = train[train.is_cis == 1].iloc[0]

print(cis_mol[['atom_index_0','atom_index_1']])
plot_molecule(cis_mol.molecule_name, structures)
trans_mol = train[train.is_trans == 1].iloc[0]

print(trans_mol[['atom_index_0','atom_index_1']])
plot_molecule(trans_mol.molecule_name, structures)
angles = pd.read_csv(f'../input/calculate-angles-and-dihedrals-with-networkx/angles.csv')
train = pd.merge(train, 

                 angles[['molecule_name','atom_index_0','atom_index_1','dihedral']],

                 how='left',

                 on=['molecule_name','atom_index_0','atom_index_1'])
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



train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
train.head()
from mpl_toolkits.mplot3d import Axes3D

from scipy import genfromtxt

import matplotlib.pyplot as plt

from pylab import rcParams

import pylab
train_3JHC = train[train['type'] == '3JHC']

train_normal = train_3JHC[(train_3JHC['is_cis'] == 0) & (train_3JHC['is_trans'] == 0)]

train_cis    = train_3JHC[train_3JHC['is_cis']   == 1]

train_trans  = train_3JHC[train_3JHC['is_trans'] == 1]
rcParams['figure.figsize'] = 10,4

train_normal['scalar_coupling_constant'].hist(bins=50)

pylab.suptitle("Not cis-trans")

plt.show()

train_cis['scalar_coupling_constant'].hist(bins=30)

pylab.suptitle("Cis")

plt.show()

train_trans['scalar_coupling_constant'].hist(bins=30)

pylab.suptitle("Trans")

plt.show()
rcParams['figure.figsize'] = 10,7



fig = plt.figure()

ax = Axes3D(fig)



ax.set_xlabel("dihedral")

ax.set_ylabel("scalar_coupling_constant")

ax.set_zlabel("dist")



ax.plot(train_normal.dihedral, 

        train_normal.dist, 

        train_normal.scalar_coupling_constant,

        "o", color='gray', ms=1, mew=0.2)

ax.plot(train_cis.dihedral,

        train_cis.dist,

        train_cis.scalar_coupling_constant,

        "o", color='blue', ms=2, mew=2)

ax.plot(train_trans.dihedral,

        train_trans.dist,

        train_trans.scalar_coupling_constant,

        "o", color='red', ms=2, mew=2)

plt.show()
train_3 = train[(train['type']=='3JHC') | (train['type']=='3JHH') | (train['type']=='3JHN')]

train_normal = train_3[(train_3['is_cis'] == 0) & (train_3['is_trans'] == 0)]

train_cis    = train_3[train_3['is_cis']   == 1]

train_trans  = train_3[train_3['is_trans'] == 1]
rcParams['figure.figsize'] = 5,7

plt.style.use('ggplot')

left = np.array(['not Cis-Trans', 'Cis', 'Trans'])

height = np.array([len(train_normal), len(train_cis), len(train_trans)])

plt.bar(left, height)
train_normal.plot(kind='scatter', x='dist', y='scalar_coupling_constant',

                  figsize=(10,4), title='not Cis and not Trans')

train_cis.plot(kind='scatter', x='dist', y='scalar_coupling_constant',

               figsize=(10,4), title='is Cis')

train_trans.plot(kind='scatter', x='dist', y='scalar_coupling_constant',

               figsize=(10,4), title='is Trans')