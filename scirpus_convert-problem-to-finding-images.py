import gc

import math

import numpy as np

import pandas as pd

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)
train = pd.read_csv('../input/train.csv')
structures = pd.read_csv('../input/structures.csv')
train.head()
structures.head()
rawimagedata = {}

sizesofmatrices = {}



for k,groupdf in tqdm((structures.groupby('molecule_name'))):

    # I am just mapping the atom types to numerics as an example feel free to one hot encode them

    groupdf.atom =  groupdf.atom.map({'H': 1, 'C': 2, 'N':3,'O':4,'F':5})

    inputimage = groupdf.merge(groupdf,on=['molecule_name'],how='outer')

    #Fermi Contact seems to love r^-3!

    inputimage['recipdistancecubed'] = 1/np.sqrt((inputimage.x_x-inputimage.x_y)**2+

                                                 (inputimage.y_x-inputimage.y_y)**2+

                                                 (inputimage.z_x-inputimage.z_y)**2)**3

    inputimage.recipdistancecubed = inputimage.recipdistancecubed.replace(np.inf,0)

    sizesofmatrices[k] = int(math.sqrt(inputimage.shape[0]))

    rawimagedata[k] = inputimage[['atom_x','atom_y','recipdistancecubed']].values.reshape(sizesofmatrices[k],sizesofmatrices[k],3)

    break
targetimages = {}

for k,groupdf in tqdm((train.groupby('molecule_name'))):



    outputimage = pd.DataFrame({'molecule_name':k,'atom_index':np.arange(sizesofmatrices[k])})

    outputimage = outputimage.merge(outputimage,on=['molecule_name'],how='outer')

    outputimage = outputimage.merge(groupdf,

                                    left_on=['molecule_name','atom_index_x','atom_index_y'],

                                    right_on=['molecule_name','atom_index_0','atom_index_1'],how='left')

    outputimage = outputimage.merge(groupdf,

                                    left_on=['molecule_name','atom_index_x','atom_index_y'],

                                    right_on=['molecule_name','atom_index_1','atom_index_0'],how='left')

    outputimage['sc'] = outputimage.scalar_coupling_constant_x.fillna(0)+outputimage.scalar_coupling_constant_y.fillna(0)

    targetimages[k] = outputimage[['sc']].values.reshape(sizesofmatrices[k],sizesofmatrices[k])

    break
rawimagedata['dsgdb9nsd_000001']
targetimages['dsgdb9nsd_000001']