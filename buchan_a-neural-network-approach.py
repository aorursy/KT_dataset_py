import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import os

print(os.listdir("../input"))
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

mulliken_charge = pd.read_csv('../input/mulliken_charges.csv')

structures = pd.read_csv('../input/structures.csv')
# Map the atom structure data into train and test files



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



def map_mulliken_charge(df,atom_idx) :

    df = pd.merge(df,mulliken_charge,how = 'left',

                 left_on = ['molecule_name',f'atom_index_{atom_idx}'],

                 right_on = ['molecule_name','atom_index']

                 )

    df = df.rename(columns={'mulliken_charge': f'mulliken_charge_{atom_idx}'}

                  )

    df = df.drop('atom_index',axis = 1)

    return df



train = map_mulliken_charge(train,0)

train = map_mulliken_charge(train,1)
potential_energy = pd.read_csv('../input/potential_energy.csv')

train = train.merge(potential_energy, on="molecule_name", how = 'inner')
dipole_moments = pd.read_csv('../input/dipole_moments.csv')

dipole_moment = np.sqrt(dipole_moments.X ** 2 + dipole_moments.Y ** 2 + dipole_moments.Z ** 2)

dipole_moments['dipole_moment'] = dipole_moment

dipole_moments = dipole_moments.drop(['X','Y','Z'],axis = 1)

train = train.merge(dipole_moments,on='molecule_name',how = 'inner')
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

from tqdm import tqdm_notebook as tqdm

atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor



fudge_factor = 0.05

atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}

print(atomic_radius)



electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}



#structures = pd.read_csv(structures, dtype={'atom_index':np.int8})



atoms = structures['atom'].values

atoms_en = [electronegativity[x] for x in tqdm(atoms)]

atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]



structures['EN'] = atoms_en

structures['rad'] = atoms_rad



display(structures.head())
i_atom = structures['atom_index'].values

p = structures[['x', 'y', 'z']].values

p_compare = p

m = structures['molecule_name'].values

m_compare = m

r = structures['rad'].values

r_compare = r



source_row = np.arange(len(structures))

max_atoms = 28



bonds = np.zeros((len(structures)+1, max_atoms+1), dtype=np.int8)

bond_dists = np.zeros((len(structures)+1, max_atoms+1), dtype=np.float32)



print('Calculating the bonds')



for i in tqdm(range(max_atoms-1)):

    p_compare = np.roll(p_compare, -1, axis=0)

    m_compare = np.roll(m_compare, -1, axis=0)

    r_compare = np.roll(r_compare, -1, axis=0)

    

    mask = np.where(m == m_compare, 1, 0) #Are we still comparing atoms in the same molecule?

    dists = np.linalg.norm(p - p_compare, axis=1) * mask

    r_bond = r + r_compare

    

    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

    

    source_row = source_row

    target_row = source_row + i + 1 #Note: Will be out of bounds of bonds array for some values of i

    target_row = np.where(np.logical_or(target_row > len(structures), mask==0), len(structures), target_row) #If invalid target, write to dummy row

    

    source_atom = i_atom

    target_atom = i_atom + i + 1 #Note: Will be out of bounds of bonds array for some values of i

    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==0), max_atoms, target_atom) #If invalid target, write to dummy col

    

    bonds[(source_row, target_atom)] = bond

    bonds[(target_row, source_atom)] = bond

    bond_dists[(source_row, target_atom)] = dists

    bond_dists[(target_row, source_atom)] = dists



bonds = np.delete(bonds, axis=0, obj=-1) #Delete dummy row

bonds = np.delete(bonds, axis=1, obj=-1) #Delete dummy col

bond_dists = np.delete(bond_dists, axis=0, obj=-1) #Delete dummy row

bond_dists = np.delete(bond_dists, axis=1, obj=-1) #Delete dummy col



print('Counting and condensing bonds')



bonds_numeric = [[i for i,x in enumerate(row) if x] for row in tqdm(bonds)]

bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate(tqdm(bond_dists))]

bond_lengths_mean = [ np.mean(x) for x in bond_lengths]

n_bonds = [len(x) for x in bonds_numeric]





bond_data = {'n_bonds':n_bonds, 'bond_lengths_mean': bond_lengths_mean }

bond_df = pd.DataFrame(bond_data)

structures = structures.join(bond_df)

display(structures.head(20))

train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)
train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

train['dist_x'] = (train['x_0'] - train['x_1']) ** 2

test['dist_x'] = (test['x_0'] - test['x_1']) ** 2

train['dist_y'] = (train['y_0'] - train['y_1']) ** 2

test['dist_y'] = (test['y_0'] - test['y_1']) ** 2

train['dist_z'] = (train['z_0'] - train['z_1']) ** 2

test['dist_z'] = (test['z_0'] - test['z_1']) ** 2



train['type_0'] = train['type'].apply(lambda x: x[0])

test['type_0'] = test['type'].apply(lambda x: x[0])
train
import gc

del structures,mulliken_charge,dipole_moments,potential_energy,dipole_moment

gc.collect()
molecules = train.pop('molecule_name')

test = test.drop('molecule_name', axis=1)



scalar_coupling_constant = train.pop('scalar_coupling_constant')

potential_energy = train.pop('potential_energy')

mulliken_charge_0 = train.pop('mulliken_charge_0')

mulliken_charge_1 = train.pop('mulliken_charge_1')

dipole_moment = train.pop('dipole_moment')





for f in ['atom_1', 'type_0', 'type','atom_0']:

        lbl = LabelEncoder()

        lbl.fit(list(train[f].values) + list(test[f].values))

        train[f] = lbl.transform(list(train[f].values))

        test[f] = lbl.transform(list(test[f].values))
train = train.drop('id',axis = 1)

test = test.drop('id',axis = 1)
scaler = preprocessing.StandardScaler()

train = scaler.fit_transform(train)
import tensorflow as tf

import keras.backend as K

from keras import metrics



import keras

from keras.engine.input_layer import Input



import matplotlib.pyplot as plt

import seaborn as sns



import random, os, sys

from keras.models import *

from keras.layers import *

from keras.callbacks import *

from keras.initializers import *

import tensorflow as tf

from keras.engine.topology import Layer

from keras import callbacks



pd.set_option('precision', 30)

np.set_printoptions(precision = 30)



np.random.seed(368)

tf.set_random_seed(368)
def nn_model() :

    i  = Input(shape = (24,))

    

    # Initial Block

    x  = Dense(64,activation = 'relu')(i)

    x  = BatchNormalization()(x)

    x  = Dense(32,activation = 'relu')(x)

    x  = BatchNormalization()(x)

    x  = Dense(16,activation = 'relu')(x)

    x  = BatchNormalization()(x)

    



    

    # Mulliken Charge 0 Block

    x1 = Dense(64,activation = 'relu')(i)

    x1 = BatchNormalization()(x1)

    x1 = Dense(32,activation = 'relu')(x1)

    x1 = BatchNormalization()(x1)

    x1 = Dense(16,activation = 'relu')(x1)

    x1 = BatchNormalization()(x1)



    x1_output = Dense(1,activation = 'linear',name = 'mulliken_charge_0')(x1)

    

    

    # Mulliken Charge 1 Block

    x2 = Dense(64,activation = 'relu')(i)

    x2 = BatchNormalization()(x2)

    x2 = Dense(32,activation = 'relu')(x2)

    x2 = BatchNormalization()(x2)

    x2 = Dense(16,activation = 'relu')(x2)

    x2 = BatchNormalization()(x2)  

    

    x2_output = Dense(1,activation = 'linear',name = 'mulliken_charge_1')(x2)

    

    # Dipole Moment Block

    x3 = Dense(128,activation = 'relu')(i)

    x3 = BatchNormalization()(x3)

    x3 = Dense(64,activation = 'relu')(x3)

    x3 = BatchNormalization()(x3)

    x3 = Dense(32,activation = 'relu')(x3)

    x3 = BatchNormalization()(x3)

    x3 = Dense(16,activation = 'relu')(x3)

    x3 = BatchNormalization()(x3)

    

    x3_output = Dense(1,activation = 'linear',name = 'dipole_moment')(x3)

    

    concat = concatenate([x,x1_output,x2_output,x3_output])

    

    # Scalar Coupling Constant Block

    x4 = Dense(64,activation = 'relu')(concat)

    x4 = BatchNormalization()(x4)

    x4 = Dense(32,activation = 'relu')(x4)

    x4 = BatchNormalization()(x4)

    x4 = Dense(16,activation = 'relu')(x4)

    x4 = BatchNormalization()(x4)

    

    x4_output = Dense(1,activation = 'linear',name = 'scaler_coupling_constant')(x4)

    

    

    return Model(inputs = [i] , outputs = [x4_output,x3_output,x2_output,x1_output])
model = nn_model()

model.compile(loss='mean_absolute_error', optimizer='adam')

model.summary()
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model,show_shapes = True).create(prog='dot', format='svg'))
history = model.fit(x = train,y = [scalar_coupling_constant.values,dipole_moment.values,mulliken_charge_1.values,mulliken_charge_0.values],

                    validation_split=0.1,epochs=100,verbose=1,batch_size = 1024)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper left')
plt.plot(history.history['scaler_coupling_constant_loss'])

plt.plot(history.history['val_scaler_coupling_constant_loss'])

plt.title('scaler_coupling_constant_loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper left')
plt.plot(history.history['dipole_moment_loss'])

plt.plot(history.history['val_dipole_moment_loss'])

plt.title('dipole_moment_loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper left')
plt.plot(history.history['mulliken_charge_1_loss'])

plt.plot(history.history['val_mulliken_charge_1_loss'])

plt.title('mulliken_charge_1_loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper left')
plt.plot(history.history['mulliken_charge_0_loss'])

plt.plot(history.history['val_mulliken_charge_0_loss'])

plt.title('mulliken_charge_0_loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper left')
y_preds = model.predict(test)
sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='id')
predictions = sample_submission.copy()

predictions['scalar_coupling_constant'] = y_preds[0]

predictions.to_csv('submission.csv')