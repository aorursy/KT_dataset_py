import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/train.csv')

train.head()
def read_xyz(path, filename):

    return pd.read_csv(path+filename, skiprows = 2, header = None, sep = ' ', usecols=[0, 1,2,3], names=['atom', 'x', 'y', 'z'])



path = '../input/structures/'

filename = 'dsgdb9nsd_000001.xyz'



read_xyz(path, filename)
# This is the code, but it is quite time consuming to run so I'll just provide the answer below

"""

atom_list = []

for filename in os.listdir("../input/structures"):

    atom_list = atom_list + list(read_xyz(path, filename)['atom'])

atom_list = set(atom_list)

print(atom_list)

"""

print("{'O', 'H', 'C', 'F', 'N'}")

x_list = []

y_list = []

z_list = []

for filename in os.listdir("../input/structures"):

    x_list = x_list + list(read_xyz(path, filename)['x'])

    y_list = y_list + list(read_xyz(path, filename)['y'])

    z_list = z_list + list(read_xyz(path, filename)['z'])

dimfig, dimaxes = plt.subplots(3, 1, figsize = (6, 6))

sns.distplot(x_list, ax=dimaxes[0])

sns.distplot(y_list, ax=dimaxes[1])

sns.distplot(z_list, ax=dimaxes[2])

print("x max: " + str(np.max(x_list)) + " x min : " + str(np.min(x_list)))

print("y max: " + str(np.max(y_list)) + " y min : " + str(np.min(y_list)))

print("z max: " + str(np.max(z_list)) + " z min : " + str(np.min(z_list)))
coupling_types = set(train['type'])

print(coupling_types)
coupling_types = list(coupling_types)

totals = [np.sum(train['type'] == x) for x in coupling_types]



subsets = dict()

for x in coupling_types:

    subsets[x] = train.loc[train['type'] == x]



bar_fig, bar_axis = plt.subplots()



sns.barplot(coupling_types, totals, ax = bar_axis)



dist_fig, dist_axes = plt.subplots(len(subsets), 1, figsize = (6, 12))



for (x, y) in zip(dist_axes, coupling_types):

    sns.distplot(subsets[y]['scalar_coupling_constant'], ax=x)

    x.set_title(y)



dist_fig.tight_layout()
def length(data, index1, index2):

    """Takes an xyz file imported by read_xyz and calculates the distance between two points"""

    return np.sqrt(np.sum(np.square(data[['x', 'y', 'z']].loc[index1]-data[['x', 'y', 'z']].loc[index2])))



def neighbours(data, index):

    """Takes an xyz file imported by read_xyz and calculates the number of neighbours within sqrt(3) Å of the indexed atom"""

    l2 = np.array([np.sum(np.square(data[['x', 'y', 'z']].loc[index]-data[['x', 'y', 'z']].loc[x])) for x in range(len(data))])

    return np.sum(l2 < 3) - 1



def nearest(data, index):

    """Takes an xyz file imported by read_xyz and finds the index of the nearest atom"""

    #data['index'] = data.index

    point = data.loc[index][['x', 'y', 'z']]

    data = data[data['atom'] != 'H'][['x', 'y', 'z']]

    data[['x', 'y', 'z']] = data[['x', 'y', 'z']] - point

    data[['x', 'y', 'z']] = np.square(data[['x', 'y', 'z']])

    data = np.sum(data, axis = 1)

    if index in data.index: data[index] = 999

    return np.argmin(data)



def magnitude(vector):

    """Calculates the magnitude of a vector"""

    return np.sqrt(np.sum(np.square(vector)))

    

def dihedral(point1, point2, point3, point4):

    """Calculates the dihederal angle between two bonds"""

    b1 = point1-point2

    b2 = point2-point3

    b3 = point3-point4

    n1 = np.cross(b1, b2)

    n1 = n1/magnitude(n1)

    n2 = np.cross(b2, b3)

    n2 = n2/magnitude(n2)

    m1 = np.cross(n1, b2/magnitude(b2))

    x = np.dot(n1, n2)

    y = np.dot(m1, n2)

    return np.arctan2(x, y)
def single_bond(coupling_type):    

    feature_list = []

    

    for x in range(1000):#len(subsets[coupling_type])):

        current = subsets[coupling_type].iloc[x]

        index0 = current['atom_index_0']

        index1 = current['atom_index_1']

        filename = current['molecule_name'] + '.xyz'

        data = read_xyz(path, filename)

        feature_list.append((length(data, index0, index1), neighbours(data, index1), current['scalar_coupling_constant']))

    

    return pd.DataFrame(feature_list, columns = ['length', 'hybrid', 'coupling'])



def two_bond(coupling_type):

    feature_list = []

    for x in range(1000):

        current = subsets[coupling_type].iloc[x]

        data = read_xyz(path, current['molecule_name'] + '.xyz')

        index_0 = current['atom_index_0']

        index_1 = current['atom_index_1']

        shared = nearest(data, index_0)

        length1 = length(data, index_0, shared)

        length2 = length(data, index_1, shared)

        vector1 = data[['x', 'y', 'z']].loc[index_0]-data[['x', 'y', 'z']].loc[shared]

        vector2 = data[['x', 'y', 'z']].loc[index_1]-data[['x', 'y', 'z']].loc[shared]

        cosine = np.dot(vector1, vector2)/(length1 * length2)

        shared_hybrid = neighbours(data, shared)

        carbon_hybrid = neighbours(data, index_1)

        feature_list.append((length1, length2, cosine, data['atom'].iloc[shared], shared_hybrid, carbon_hybrid, current['scalar_coupling_constant']))

    return pd.DataFrame(feature_list, columns = ['length1', 'length2', 'cosine', 'atom', 'hybrid1', 'hybrid2', 'coupling'])



def three_bond(coupling_type):

    feature_list = []

    for x in range(1000):

        current = subsets[coupling_type].iloc[x]

        data = read_xyz(path, current['molecule_name'] + '.xyz')

        index_0 = current['atom_index_0']

        index_1 = current['atom_index_1']

        shared1 = nearest(data, index_0)

        shared2 = nearest(data, index_1)

        length1 = length(data, index_0, shared1)

        length2 = length(data, index_1, shared2)

        length_shared = length(data, index_0, index_1)

        cosine = dihedral(data[['x', 'y', 'z']].loc[index_0], data[['x', 'y', 'z']].loc[shared1], data[['x', 'y', 'z']].loc[shared2], data[['x', 'y', 'z']].loc[index_1])

        shared1_hybrid = neighbours(data, shared1)

        shared2_hybrid = neighbours(data, shared2)

        terminal_hybrid = neighbours(data, index_1)

        feature_list.append((length1, length2, length_shared, cosine, data['atom'].iloc[shared1], data['atom'].iloc[shared2], shared1_hybrid, shared2_hybrid, terminal_hybrid, current['scalar_coupling_constant']))

    return pd.DataFrame(feature_list, columns = ['length1', 'length2', 'length_shared', 'angle', 'atom1', 'atom2', 'hybrid1', 'hybrid2', 'terminal_hybrid', 'coupling'])



function_dict = {'1': single_bond, '2': two_bond, '3': three_bond}

engineered = {x:function_dict[x[0]](x) for x in coupling_types}
sns.scatterplot(engineered['3JHH']['angle'], engineered['3JHH']['coupling'], hue=engineered['3JHH']['length_shared'])
sns.pairplot(engineered['3JHH'])
sns.pairplot(engineered['3JHC'])
sns.pairplot(engineered['3JHN'])
sns.pairplot(engineered['2JHH'])
sns.pairplot(engineered['2JHC'])
sns.pairplot(engineered['2JHN'])
sns.pairplot(engineered['1JHC'])
sns.pairplot(engineered['1JHN'])