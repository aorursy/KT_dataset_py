# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_original = pd.read_csv("../input/train.csv")

structures_original = pd.read_csv("../input/structures.csv")

test_original = pd.read_csv("../input/test.csv")
train_original.head()
structures_original.head()
structures_original[structures_original['molecule_name'] == 'dsgdb9nsd_000015']
moleculeCount = structures_original.groupby(by=['molecule_name','atom'])[['atom']].count()

moleculeCount.rename(columns={'atom':'count'},inplace = True)

moleculeCount = moleculeCount.unstack(fill_value=0)

moleculeCount = moleculeCount['count'].reset_index()



moleculeCount.head()
moleculeCount[moleculeCount['molecule_name'] == 'dsgdb9nsd_000015']
structures = pd.DataFrame.merge(structures_original,moleculeCount

                               ,how='inner'

                               ,left_on = ['molecule_name'] 

                               ,right_on = ['molecule_name']

                              )



structures.head()
tmp_merge = pd.DataFrame.merge(train_original,structures

                               ,how='left'

                               ,left_on = ['molecule_name','atom_index_0'] 

                               ,right_on = ['molecule_name','atom_index']

                              )



tmp_merge = tmp_merge.merge(structures

                ,how='left'

                ,left_on = ['molecule_name','atom_index_1'] 

                ,right_on = ['molecule_name','atom_index']

               )



tmp_merge.drop(columns=['atom_index_x','atom_index_y','C_x','F_x','H_x','N_x','O_x'],inplace=True)

tmp_merge.columns = ['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type' , 'scalar_coupling_constant' , 

                      'atom_nm_0' , 'x_0' , 'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']



train = tmp_merge[['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type'  , 'atom_nm_0' , 'x_0' ,

           'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O', 'scalar_coupling_constant']]

train.sort_values(by=['id','molecule_name'],inplace=True)

train.reset_index(inplace=True,drop=True)



tmp_merge = None



train['dist'] = np.linalg.norm(train[['x_0', 'y_0', 'z_0']].values - train[['x_1', 'y_1', 'z_1']].values, axis=1)

train.drop(columns=['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'],inplace=True)



train.head()
tmp_merge = pd.DataFrame.merge(test_original,structures

                               ,how='left'

                               ,left_on = ['molecule_name','atom_index_0'] 

                               ,right_on = ['molecule_name','atom_index']

                              )



tmp_merge = tmp_merge.merge(structures

                ,how='left'

                ,left_on = ['molecule_name','atom_index_1'] 

                ,right_on = ['molecule_name','atom_index']

               )



tmp_merge.drop(columns=['atom_index_x','atom_index_y','C_x','F_x','H_x','N_x','O_x'],inplace=True)

tmp_merge.columns = ['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type' , 

                      'atom_nm_0' , 'x_0' , 'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']



test = tmp_merge[['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type'  , 'atom_nm_0' , 'x_0' ,

           'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']]

test.sort_values(by=['id','molecule_name'],inplace=True)

test.reset_index(inplace=True,drop=True)



tmp_merge = None



test['dist'] = np.linalg.norm(test[['x_0', 'y_0', 'z_0']].values - test[['x_1', 'y_1', 'z_1']].values, axis=1)

test.drop(columns=['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'],inplace=True)



test.head()