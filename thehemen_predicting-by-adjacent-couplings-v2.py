# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import math

import tqdm

import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



os.listdir('/kaggle')
class Coupling:

    def __init__(self, index, atom_index_0, atom_index_1, coupling_type, scalar_coupling_constant=None):

        self.index = index

        self.atom_index_0 = atom_index_0

        self.atom_index_1 = atom_index_1

        self.coupling_type = coupling_type

        self.scalar_coupling_constant = scalar_coupling_constant



class Atom:

    def __init__(self, index, name, x, y, z):

        self.index = index

        self.name = name

        self.x = x

        self.y = y

        self.z = z



        self.coupling_ids = []



class Molecule:

    def __init__(self, name):

        self.name = name

        self.atoms = {}

        self.couplings = []



    def add_coupling(self, coupling):

        self.couplings.append(coupling)

        self.atoms[coupling.atom_index_0].coupling_ids.append(len(self.couplings) - 1)

        self.atoms[coupling.atom_index_1].coupling_ids.append(len(self.couplings) - 1)



def get_molecule_index_from_name(name):

    return int(name.split('_')[-1])
def get_structures(filename):

    molecules = {}

    atom_names = []



    with open(filename, 'r') as f:

        structures_df = pd.read_csv(f)

        atom_names = list(structures_df['atom'].unique())



        with tqdm.tqdm(total=structures_df.shape[0]) as tqdm_bar:

            for i, row in structures_df.iterrows():

                molecule_index = get_molecule_index_from_name(row.values[0])

                atom_index = row.values[1]

                atom_name = row.values[2]

                x = row.values[3]

                y = row.values[4]

                z = row.values[5]



                if molecule_index not in molecules.keys():

                    molecules[molecule_index] = Molecule(molecule_index)



                molecules[molecule_index].atoms[atom_index] = Atom(atom_index, atom_name, x, y, z)

                tqdm_bar.update(1)



    return molecules, atom_names



print('Reading structures...')

molecules, atom_names = get_structures('/kaggle/input/champs-scalar-coupling/structures.csv')

molecule_count = len(molecules)

print('Molecules: {}'.format(molecule_count))

print('Atom names: {}'.format(atom_names))
def get_train_data(filename):

    train_molecule_indexes = []



    with open(filename, 'r') as f:

        train_df = pd.read_csv(f)

        coupling_types = list(train_df['type'].unique())



        with tqdm.tqdm(total=train_df.shape[0]) as tqdm_bar:

            for i, row in train_df.iterrows():

                coupling_index = row.values[0]

                molecule_index = get_molecule_index_from_name(row.values[1])

                atom_index_0 = row.values[2]

                atom_index_1 = row.values[3]

                coupling_type = row.values[4]

                scalar_coupling_constant = row.values[5]



                coupling = Coupling(coupling_index, atom_index_0, atom_index_1, coupling_type,

                    scalar_coupling_constant)

                molecules[molecule_index].add_coupling(coupling)

                train_molecule_indexes.append(molecule_index)

                tqdm_bar.update(1)



    train_molecule_indexes = sorted(list(set(train_molecule_indexes)))

    return train_molecule_indexes, coupling_types



print('Reading train data...')

train_molecule_indexes, coupling_types = get_train_data('/kaggle/input/champs-scalar-coupling/train.csv')

train_molecule_count = len(train_molecule_indexes)

print('Train molecules: {}'.format(train_molecule_count))

print('Coupling types: {}'.format(coupling_types))
def get_test_data(filename):

    test_molecule_indexes = []



    with open(filename, 'r') as f:

        test_df = pd.read_csv(f)

        coupling_types = list(test_df['type'].unique())



        with tqdm.tqdm(total=test_df.shape[0]) as tqdm_bar:

            for i, row in test_df.iterrows():

                coupling_index = row.values[0]

                molecule_index = get_molecule_index_from_name(row.values[1])

                atom_index_0 = row.values[2]

                atom_index_1 = row.values[3]

                coupling_type = row.values[4]



                coupling = Coupling(coupling_index, atom_index_0, atom_index_1, coupling_type)

                molecules[molecule_index].add_coupling(coupling)

                test_molecule_indexes.append(molecule_index)

                tqdm_bar.update(1)



    test_molecule_indexes = sorted(list(set(test_molecule_indexes)))

    return test_molecule_indexes



print('Reading test data...')

test_molecule_indexes = get_test_data('/kaggle/input/champs-scalar-coupling/test.csv')

test_molecule_count = len(test_molecule_indexes)

print('Test molecules: {}'.format(test_molecule_count))
max_adjacent_couplings = 9

train_coupling_count = 0

test_coupling_count = 0



for molecule_index, molecule in molecules.items():

    if molecule_index in train_molecule_indexes:

        train_coupling_count += len(molecule.couplings)

    else:

        test_coupling_count += len(molecule.couplings)



print('Max adjacent couplings: {}'.format(max_adjacent_couplings))

print('Train couplings count: {}'.format(train_coupling_count))

print('Test couplings count: {}'.format(test_coupling_count))
one_input_dim = 7

side_input_count = max_adjacent_couplings + 1



X_train = []

for i in range(2):

    for j in range(side_input_count):

        X_train.append(np.zeros(shape=(train_coupling_count, one_input_dim), dtype=np.float))



Y_train = np.zeros(shape=(train_coupling_count), dtype=np.float)



X_test = []

for i in range(2):

    for j in range(side_input_count):

        X_test.append(np.zeros(shape=(test_coupling_count, one_input_dim), dtype=np.float))



def get_distance(x0, y0, z0, x1, y1, z1):

    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)



def get_angle_cos(x0, y0, z0, x1, y1, z1):

    return (x0 * x1 + y0 * y1 + z0 * z1) / math.sqrt((x0 ** 2 + y0 ** 2 + z0 ** 2) * (x1 ** 2 + y1 ** 2 + z1 ** 2))
def make_dataset(X_arr, molecule_indexes, total_coupling_count, with_output):

    i = 0



    with tqdm.tqdm(total=total_coupling_count) as tqdm_bar:

        for molecule_index in molecule_indexes:

            molecule = molecules[molecule_index]

            coupling_count = len(molecule.couplings)



            for coupling_index in range(coupling_count):

                atom_index_0 = molecule.couplings[coupling_index].atom_index_0

                atom_index_1 = molecule.couplings[coupling_index].atom_index_1



                c_x = molecule.atoms[atom_index_1].x - molecule.atoms[atom_index_0].x

                c_y = molecule.atoms[atom_index_1].y - molecule.atoms[atom_index_0].y

                c_z = molecule.atoms[atom_index_1].z - molecule.atoms[atom_index_0].z



                if with_output:

                    Y_train[i] = molecule.couplings[coupling_index].scalar_coupling_constant



                for side_index, atom_index in enumerate([ atom_index_0, atom_index_1 ]):

                    coupling_type = molecule.couplings[coupling_index].coupling_type

                    atom_name = molecule.atoms[atom_index].name



                    x0 = molecule.atoms[atom_index].x

                    y0 = molecule.atoms[atom_index].y

                    z0 = molecule.atoms[atom_index].z



                    j = 0

                    X_arr[side_index * side_input_count + j][i, 0] = coupling_types.index(coupling_type)

                    X_arr[side_index * side_input_count + j][i, 1] = atom_names.index(atom_name)

                    X_arr[side_index * side_input_count + j][i, 2] = x0 - c_x

                    X_arr[side_index * side_input_count + j][i, 3] = y0 - c_y

                    X_arr[side_index * side_input_count + j][i, 4] = z0 - c_z

                    X_arr[side_index * side_input_count + j][i, 5] = get_distance(c_x, c_y, c_z, x0, y0, z0)

                    X_arr[side_index * side_input_count + j][i, 6] = get_angle_cos(c_x, c_y, c_z, x0, y0, z0)



                    distances = []



                    for coupling_index_to in molecule.atoms[atom_index].coupling_ids:

                        if coupling_index != coupling_index_to:

                            _atom_index_0 = molecule.couplings[coupling_index_to].atom_index_0

                            _atom_index_1 = molecule.couplings[coupling_index_to].atom_index_1



                            atom_index_to = 0



                            if atom_index != _atom_index_0:

                                atom_index_to = _atom_index_0

                            else:

                                atom_index_to = _atom_index_1



                            x1 = molecule.atoms[atom_index_to].x

                            y1 = molecule.atoms[atom_index_to].y

                            z1 = molecule.atoms[atom_index_to].z



                            distance = get_distance(x0, y0, z0, x1, y1, z1)

                            angle = get_angle_cos(x0, y0, z0, x1, y1, z1)

                            distances.append(tuple((coupling_index_to, atom_index_to, distance, angle)))



                    distances.sort(key=lambda x: x[2])



                    j += 1

                    for coupling_index_to, atom_index_to, distance, angle in distances[:max_adjacent_couplings]:

                        atom_name = molecule.atoms[atom_index_to].name

                        coupling_type = molecule.couplings[coupling_index_to].coupling_type



                        x1 = molecule.atoms[atom_index_to].x

                        y1 = molecule.atoms[atom_index_to].y

                        z1 = molecule.atoms[atom_index_to].z



                        X_arr[side_index * side_input_count + j][i, 0] = coupling_types.index(coupling_type)

                        X_arr[side_index * side_input_count + j][i, 1] = atom_names.index(atom_name)

                        X_arr[side_index * side_input_count + j][i, 2] = x1 - c_x

                        X_arr[side_index * side_input_count + j][i, 3] = y1 - c_y

                        X_arr[side_index * side_input_count + j][i, 4] = z1 - c_z

                        X_arr[side_index * side_input_count + j][i, 5] = distance

                        X_arr[side_index * side_input_count + j][i, 6] = angle

                        j += 1



                i += 1

                tqdm_bar.update(1)
print('Creating train dataset...')

make_dataset(X_train, train_molecule_indexes, train_coupling_count, with_output=True)

print('X_train: {} x {}'.format(X_train[0].shape, len(X_train)))

print('Y_train: {}'.format(Y_train.shape))
print('Creating test dataset...')

make_dataset(X_test, test_molecule_indexes, test_coupling_count, with_output=False)

print('X_test: {} x {}'.format(X_test[0].shape, len(X_test)))
from keras.models import Model

from keras.layers import *

from keras.callbacks import EarlyStopping, ReduceLROnPlateau



def get_model(input_dims):

    input_now = Input(shape=(input_dims[0],))

    x = Dense(48)(input_now)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dense(16)(x)

    x = LeakyReLU(alpha=0.05)(x)

    model_one = Model(inputs=input_now, outputs=x, name='model_one')



    input_now = Input(shape=(input_dims[1],))

    x = Dense(48)(input_now)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dense(16)(x)

    x = LeakyReLU(alpha=0.05)(x)

    model_two = Model(inputs=input_now, outputs=x, name='model_two')



    input_now = Input(shape=(160,))

    x = Dense(1024)(input_now)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dense(512)(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dense(256)(x)

    x = LeakyReLU(alpha=0.05)(x)

    model_side = Model(inputs=input_now, outputs=x, name='model_side')



    inputs = []

    sides = []



    for i, side_dims in enumerate([ input_dims[:len(input_dims) // 2], input_dims[len(input_dims) // 2:] ]):

        nodes = []



        input_now = Input(shape=(side_dims[0],), name='input_0_{}'.format(i))

        x = model_one(input_now)

        x = BatchNormalization(name='batchnorm_0_{}'.format(i))(x)

        inputs.append(input_now)

        nodes.append(x)



        for j, input_dim in enumerate(side_dims[1:]):

            input_now = Input(shape=(input_dim,), name='input_1_{}_{}'.format(i, j))

            x = model_two(input_now)

            x = BatchNormalization(name='batchnorm_1_{}_{}'.format(i, j))(x)

            inputs.append(input_now)

            nodes.append(x)



        x = concatenate(nodes)

        x = model_side(x)

        x = BatchNormalization(name='batchnorm_2_{}'.format(i))(x)

        sides.append(x)



    x = concatenate(sides)

    x = Dense(256, name='dense_0')(x)

    x = BatchNormalization(name='batchnorm_3')(x)

    x = LeakyReLU(alpha=0.05, name='leaky_relu_0')(x)

    x = Dense(128, name='dense_1')(x)

    x = BatchNormalization(name='batchnorm_4')(x)

    x = LeakyReLU(alpha=0.05, name='leaky_relu_1')(x)

    x = Dense(32, name='dense_2')(x)

    x = BatchNormalization(name='batchnorm_5')(x)

    x = LeakyReLU(alpha=0.05, name='leaky_relu_2')(x)

    output = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer='adam', loss='mean_absolute_error')

    return model



input_dims = [ X_val.shape[1] for X_val in X_train ]

model = get_model(input_dims)

model.summary()
epochs = 250

batch_size = 2048

val_split = 0.1



early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=40, verbose=1, mode='auto',

    restore_best_weights=True)

reduce_on_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, min_lr=1e-6, mode='auto',

    verbose=1)



hist = model.fit(X_train, Y_train, validation_split=val_split, batch_size=batch_size, epochs=epochs,

    callbacks=[ early_stopping, reduce_on_lr ], verbose=1)
import matplotlib.pyplot as plt



def plot_hist(hist, filename):

    plt.plot(hist.history['loss'])

    plt.plot(hist.history['val_loss'])

    plt.title('Mean Absolute Error')

    plt.ylabel('Loss')

    plt.xlabel('Epochs')

    plt.legend(['Train', 'Val'], loc='upper left')

    plt.savefig(filename)



plot_hist(hist, '/kaggle/working/plot.png')
submission_results = model.predict(X_test, batch_size=batch_size, verbose=1)[:,0].tolist()



with open('/kaggle/input/champs-scalar-coupling/sample_submission.csv', 'r') as f:

    submit = pd.read_csv(f)

    submit["scalar_coupling_constant"] = submission_results

    submit.to_csv("/kaggle/working/submission.csv", index=False)



submit.head()