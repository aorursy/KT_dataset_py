import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.decomposition import PCA

from sklearn.metrics import pairwise_distances

from sklearn.neighbors import NearestNeighbors



import os

import warnings  

print(os.listdir("../input"))
structures = pd.read_csv('../input/structures.csv') 



# uncomment for debugging 

structures = structures.head(n=100)



structures.head(n=10)


def nn_features(l):

    ''' l: indexed pd.Series of a molecule '''

    

    # number of nearest neighbors +1

    k = 4+1

    

    # lookup coordinates of atoms in molecule 

    x=np.array(structures.loc[l.index,'x'])

    y=np.array(structures.loc[l.index,'y'])

    z=np.array(structures.loc[l.index,'z'])

    coord = np.append(np.append(x,y),z).reshape((l.size,3),order='F')

    

    # NN calculations

    nbrs = NearestNeighbors(n_neighbors=min(len(coord),k), algorithm='ball_tree').fit(coord)

    distances, indices = nbrs.kneighbors(coord)

    

    

    if indices.shape != (1,1):

        # PCA - not relevant for nn, but nice feature anyway

        pca = PCA(n_components=2)

        p=pca.fit_transform(coord)

        

        # NN id and NN distance

        atm = np.pad(indices[:,1:l.size],((0,0),(0, max(0, k-l.size))), 'constant', constant_values=(999, 999))

        dst = np.pad(distances[:,1:l.size], ((0,0),(0,max(0,k-l.size))), 'constant', constant_values=(0, 0))

        

        # LookUps for atom name and x,y,z, default value N/A or 0

        lu = np.append(np.array(structures.loc[l.index,'atom']),np.array('N/A'))

        lu_x = np.append(np.array(structures.loc[l.index,'x']),np.array(0))

        lu_y = np.append(np.array(structures.loc[l.index,'y']),np.array(0))

        lu_z = np.append(np.array(structures.loc[l.index,'z']),np.array(0))

        

        # for each nn look up coordinates and atom name 

        nn_x = np.take(lu_x, atm, mode = 'clip') 

        nn_y = np.take(lu_y, atm, mode = 'clip') 

        nn_z = np.take(lu_z, atm, mode = 'clip') 

        atm = np.take(lu, atm, mode = 'clip')

    else: 

        # in case the molecule contains only 1 atom (e.g. while debugging a small dataset)

        p = np.ones((1, 2))*(999)

        atm = np.ones((1, max(0, k-l.size)))*(999) 

        dst = np.ones((1, max(0, k-l.size)))*(999)

        nn_x = np.ones((1, max(0, k-l.size)))*(999)

        nn_y = np.ones((1, max(0, k-l.size)))*(999)

        nn_z = np.ones((1, max(0, k-l.size)))*(999)

    

    # put together atom names, distances, coordinates of nnearest neighbors and pca

    out = np.append(np.append(np.append(np.append(np.append(atm,dst,axis=1),nn_x, axis=1),nn_y, axis=1),nn_z, axis=1) ,p, axis=1)

    

    return [i for i in out]



%%time



warnings.filterwarnings('ignore')



structures['nearestn'] = structures.groupby('molecule_name')['x'].transform(nn_features)



structures.head(n=10)

#11mi 12s
%%time



# atom name of nn

structures['nn_1'] = structures['nearestn'].apply(lambda x: x[0])

structures['nn_2'] = structures['nearestn'].apply(lambda x: x[1])

structures['nn_3'] = structures['nearestn'].apply(lambda x: x[2])

structures['nn_4'] = structures['nearestn'].apply(lambda x: x[3])



# eucledian distances to nn

structures['nn_1_dist'] = structures['nearestn'].apply(lambda x: x[4])

structures['nn_2_dist'] = structures['nearestn'].apply(lambda x: x[5])

structures['nn_3_dist'] = structures['nearestn'].apply(lambda x: x[6])

structures['nn_4_dist'] = structures['nearestn'].apply(lambda x: x[7])



# x,y,z distances to nn

structures['nn_dx_1'] = structures['nearestn'].apply(lambda x: x[8])  - structures['x']

structures['nn_dx_2'] = structures['nearestn'].apply(lambda x: x[9])  - structures['x']

structures['nn_dx_3'] = structures['nearestn'].apply(lambda x: x[10])  - structures['x']

structures['nn_dx_4'] = structures['nearestn'].apply(lambda x: x[11])  - structures['x']



structures['nn_dy_1'] = structures['nearestn'].apply(lambda x: x[12])  - structures['y']

structures['nn_dy_2'] = structures['nearestn'].apply(lambda x: x[13])  - structures['y']

structures['nn_dy_3'] = structures['nearestn'].apply(lambda x: x[14])  - structures['y']

structures['nn_dy_4'] = structures['nearestn'].apply(lambda x: x[15])  - structures['y']



structures['nn_dz_1'] = structures['nearestn'].apply(lambda x: x[16])  - structures['z']

structures['nn_dz_2'] = structures['nearestn'].apply(lambda x: x[17])  - structures['z']

structures['nn_dz_3'] = structures['nearestn'].apply(lambda x: x[18])  - structures['z']

structures['nn_dz_4'] = structures['nearestn'].apply(lambda x: x[19])  - structures['z']



# 2 dim pca

structures['pca_x'] = structures['nearestn'].apply(lambda x: x[20])

structures['pca_y'] = structures['nearestn'].apply(lambda x: x[21])



structures = structures.drop(columns='nearestn',axis=0)

structures.head(n=10)