import numpy as np

import pandas as pd

from tqdm import tqdm

from scipy import spatial
df = pd.read_csv('../input/train.csv')

df.head(10)
df_structures = pd.read_csv('../input/structures.csv')

df_structures.head(10)
indices = None

distances = None

for c in tqdm(df_structures.molecule_name.unique()[:5]):

    x = df_structures[df_structures.molecule_name==c][['x','y','z']]

    z = spatial.KDTree(x,x.shape[0])

    permol = z.query(x,x.shape[0])

    if(indices is None):

        distances = permol[0].tolist()

        indices = permol[1].tolist()

    else:

        distances += permol[0].tolist()

        indices += permol[1].tolist()

nearestneighbors = pd.DataFrame(indices)

distanceneighbors = pd.DataFrame(distances)
nearestneighbors.head(10)
distanceneighbors.head(10)
distances = {}

atoms = {}



i = 0

for k,groupdf in tqdm(df_structures.groupby('molecule_name')):

    

    x = groupdf[['atom']]

    z = spatial.KDTree(groupdf[['x','y','z']],x.shape[0])

    permol = z.query(groupdf[['x','y','z']],x.shape[0])

    d = np.take_along_axis(permol[0],np.argsort(permol[1]),axis=1)

    o1 = np.arange(d.shape[1]).reshape(1,-1)

    o2 = np.repeat(o1,o1.shape[1],axis=0)[np.triu_indices(o1.shape[1], k = 1)]

    o3 = np.repeat(o1,o1.shape[1],axis=0).T[np.triu_indices(o1.shape[1], k = 1)]

    indices = np.array([str(i2) +'_'+str(i1) for i1,i2 in zip(o2,o3)]).tolist()

    d = d[np.triu_indices(d.shape[0], k = 1)].tolist()

    distances[k] =  dict(zip(indices, d))

    a = x.atom.values.reshape(1,-1)

    b = np.repeat(a,a.shape[1],axis=0)[np.triu_indices(a.shape[1], k = 1)]

    c = np.repeat(a,a.shape[1],axis=0).T[np.triu_indices(a.shape[1], k = 1)]

    atoms[k] = dict(zip(indices, (b+c).tolist()))

  

    

    if(i==10):

        break

    i+=1

distanceneighbors = pd.DataFrame.from_dict(distances,orient='index')

atomneighbors = pd.DataFrame.from_dict(atoms,orient='index')
distanceneighbors.head(10)


atomneighbors.head(10)