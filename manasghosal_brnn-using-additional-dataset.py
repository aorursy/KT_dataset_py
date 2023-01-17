# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn import linear_model

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
le = LabelEncoder()

oh = OneHotEncoder(sparse = False)

df = pd.read_csv('../input/train.csv',sep=',', header=0, usecols=['molecule_name','atom_index_0','atom_index_1','type','scalar_coupling_constant'] )

df_m_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv',

                       sep=',', header=0, usecols=['molecule_name','atom_index_0','atom_index_1','fc','sd','pso', 'dso'])

df = pd.merge(df, df_m_coupling_contributions, on=['molecule_name','atom_index_0','atom_index_1'])

df_m_diple_moments = pd.read_csv('../input/dipole_moments.csv',sep=',', header=0, usecols=['molecule_name','X','Y','Z'])

df = pd.merge(df, df_m_diple_moments, on='molecule_name')

df_m_pot_engy = pd.read_csv('../input/potential_energy.csv',sep=',', header=0, usecols=['molecule_name','potential_energy'])

df = pd.merge(df, df_m_pot_engy, on='molecule_name')



df_a_str = pd.read_csv('../input/structures.csv',

                       sep=',', header=0, usecols=['molecule_name','atom_index','atom','x','y','z'])

#df_a_str['atom'] = le.fit_transform(df_a_str['atom'])

f = df_a_str['atom'].values

f = np.reshape(f, (-1,1))

f = oh.fit_transform(f)

ohdf = pd.DataFrame(f)

df_a_str = pd.concat([df_a_str, ohdf], axis=1)

df_a_str = df_a_str.rename(index=str, columns={0: 'A0',1:'A1',2:'A2',3:'A3',4:'A4'})

df_a_str.drop(columns=['atom'], inplace = True)

df_a_mag_sh_tensor = pd.read_csv('../input/magnetic_shielding_tensors.csv',

                       sep=',', header=0, usecols=['molecule_name','atom_index','XX','YX','ZX','XY','YY','ZY','XZ','YZ','ZZ'])



df_a_mlkn_charges = pd.read_csv('../input/mulliken_charges.csv',sep=',', header=0, usecols=['molecule_name','atom_index','mulliken_charge'])



df_a_str = pd.merge(df_a_str, df_a_mag_sh_tensor, on=['molecule_name','atom_index'])



df_a_str = pd.merge(df_a_str, df_a_mlkn_charges, on=['molecule_name','atom_index'])



df_atom_1_prop = df_a_str.rename(index=str, columns={'atom_index': 'atom_index_0','A0':'A0_0','A1':'A1_0','A2':'A2_0','A3':'A3_0','A4':'A4_0','x':'x_0','y':'y_0','z':'z_0', 'XX':'XX_0', 'YX':'YX_0', 'ZX':'ZX_0', 'XY':'XY_0', 'YY':'YY_0', 'ZY':'ZY_0', 'XZ':'XZ_0', 'YZ':'YZ_0', 'ZZ':'ZZ_0', 'mulliken_charge':'mulliken_charge_0'})

df = pd.merge(df, df_atom_1_prop, on=['molecule_name','atom_index_0'])



df_atom_2_prop = df_a_str.rename(index=str, columns={'atom_index': 'atom_index_1','A0':'A0_1','A1':'A1_1','A2':'A2_1','A3':'A3_1','A4':'A4_1','x':'x_1','y':'y_1','z':'z_1', 'XX':'XX_1', 'YX':'YX_1', 'ZX':'ZX_1', 'XY':'XY_1', 'YY':'YY_1', 'ZY':'ZY_1', 'XZ':'XZ_1', 'YZ':'YZ_1', 'ZZ':'ZZ_1', 'mulliken_charge':'mulliken_charge_1'})

df = pd.merge(df, df_atom_2_prop, on=['molecule_name','atom_index_1'])

ss = StandardScaler()

df[['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso', 'X', 'Y', 'Z',

       'potential_energy', 'x_0', 'y_0', 'z_0', 'A0_0', 'A1_0', 'A2_0', 'A3_0',

       'A4_0', 'XX_0', 'YX_0', 'ZX_0', 'XY_0', 'YY_0', 'ZY_0', 'XZ_0', 'YZ_0',

       'ZZ_0', 'mulliken_charge_0', 'x_1', 'y_1', 'z_1', 'A0_1', 'A1_1',

       'A2_1', 'A3_1', 'A4_1', 'XX_1', 'YX_1', 'ZX_1', 'XY_1', 'YY_1', 'ZY_1',

       'XZ_1', 'YZ_1', 'ZZ_1', 'mulliken_charge_1']] = ss.fit_transform(df[['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso', 'X', 'Y', 'Z',

       'potential_energy', 'x_0', 'y_0', 'z_0', 'A0_0', 'A1_0', 'A2_0', 'A3_0',

       'A4_0', 'XX_0', 'YX_0', 'ZX_0', 'XY_0', 'YY_0', 'ZY_0', 'XZ_0', 'YZ_0',

       'ZZ_0', 'mulliken_charge_0', 'x_1', 'y_1', 'z_1', 'A0_1', 'A1_1',

       'A2_1', 'A3_1', 'A4_1', 'XX_1', 'YX_1', 'ZX_1', 'XY_1', 'YY_1', 'ZY_1',

       'XZ_1', 'YZ_1', 'ZZ_1', 'mulliken_charge_1']])
def build_atom_pairs(name, molecule):

    df = molecule.apply(list)

    atom_pair_y = np.zeros((df.shape[0], 8))

    

    atom_pair = np.zeros((df.shape[0], 2, 18))

    atom_pair[:,0,:] = df.as_matrix(columns=['x_0','y_0','z_0','XX_0','YX_0','ZX_0','XY_0',

                                            'YY_0','ZY_0','XZ_0','YZ_0','ZZ_0','mulliken_charge_0',

                                             'A0_0','A1_0','A2_0','A3_0','A4_0'])

    atom_pair[:,1,:] = df.as_matrix(columns=['x_1','y_1','z_1','XX_1','YX_1','ZX_1','XY_1',

                                            'YY_1','ZY_1','XZ_1','YZ_1','ZZ_1','mulliken_charge_1',

                                             'A0_1','A1_1','A2_1','A3_1','A4_1'])

   

    atom_pair_y = df.as_matrix(columns=['potential_energy','X','Y','Z','fc','sd','pso','dso'])

    return atom_pair, atom_pair_y
moleculelist = []

molecule_ylist = []
molecules = df.groupby('molecule_name')

c = 0

for name, molecule in molecules:

    atoms, molecule_y = build_atom_pairs(name, molecule)

    amolecule = np.zeros((650,atoms.shape[1], atoms.shape[2]))

    amolecule[:atoms.shape[0],:atoms.shape[1],:atoms.shape[2]] = atoms

    amolecule = amolecule.transpose([0,2,1]).reshape(amolecule.shape[0], -1)

    amolecule_y = np.zeros((650,molecule_y.shape[1]))

    amolecule_y[:molecule_y.shape[0],:molecule_y.shape[1]] = molecule_y

    moleculelist.append(amolecule)

    molecule_ylist.append(amolecule_y)

    c = c + 1

    if c > 10000:

        break
from keras import layers

from keras.optimizers import Adam

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten

from keras.layers import Dropout

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.models import Sequential

from keras.layers import Bidirectional

from keras.layers import LSTM
def BRNNModel(inputdim):

    model = Sequential()

    model.add(Bidirectional(LSTM(100, return_sequences=True, input_dim= inputdim )))

    #model.add(Bidirectional(LSTM(10)))

    model.add(Dense(8))

    model.add(Activation('relu'))

    return model
def batch_generator(X, y, batch_size):

    number_of_batches = X.shape[0]/batch_size

    counter=0

    shuffle_index = np.arange(np.shape(y)[0])

#    X =  X[shuffle_index, :]

#    y =  y[shuffle_index]

    while 1:

        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]

        X_batch = X[index_batch,:].todense()

        y_batch = y[index_batch]

        counter += 1

        yield(np.array(X_batch),y_batch)

        if (counter > number_of_batches):

#            np.random.shuffle(shuffle_index)

            counter=0
X = np.asarray(moleculelist)

y = np.asarray(molecule_ylist)
X_train, X_test, Y_train, Y_test = train_test_split(X, y)
model = BRNNModel(X.shape[1])

model.compile(optimizer = 'adam', loss = "mean_squared_error", metrics = ["accuracy"])#"adam"

model.fit(X_train, Y_train, epochs=30, batch_size=16, verbose=2)
#model.fit_generator(generator=batch_generator(X_train, Y_train, batch_size=32), 

#                    nb_epoch=10, samples_per_epoch=X_train.shape[0])
preds = model.evaluate(x = X_test, y = Y_test)

print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))