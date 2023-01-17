# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings

warnings.filterwarnings('ignore')
import numpy as np

from sklearn.model_selection import train_test_split

from random import randrange

import pandas as pd

import lightgbm as lgb

import matplotlib.pyplot as plt

import seaborn as sns
data = np.load('/kaggle/input/pair_data.npy', allow_pickle=True)

data_length = data.__len__()
atoms = np.zeros(shape=(data_length, 32))

envs = np.zeros(shape=(data_length, 32))

freqs = np.zeros(shape=data_length)



for i in range(data.__len__()):

    x = data[i]

    atoms[i] = x['center']

    envs[i] = x['env']

    freqs[i] = x['freq']
def atom_hash(atom):

    hash = 0

    for i in range(atom.__len__()):

        hash += atom[i] * 2**i

    return hash
set_atom_hash = set()

for i in range(atoms.__len__()):    

    set_atom_hash.add(atom_hash(atoms[i]))

set_atom_hash.__len__()
set_env_hash = set()

for i in range(envs.__len__()):    

    set_env_hash.add(atom_hash(envs[i]))

set_env_hash.__len__()
atom_train, atom_test, env_train, env_test = train_test_split(atoms, envs, test_size=0.1, random_state=42)
label_1 = np.concatenate((env_train, atom_train), axis=1)
def generate_label_0(atom_data, env_data, num_label=4):

    data_length = atom_data.__len__()

    label_0_data = []

    for i in range(atom_data.__len__()):

        count_random_index = 0

        while count_random_index < num_label:

            random_index = randrange(data_length)

            if random_index != i:                

                label_0_data.append(np.concatenate((env_data[i], atom_data[int(random_index)])))

                count_random_index += 1

    return label_0_data
label_0 = generate_label_0(atom_train, env_train)

label_0_ndarray = np.array(label_0)
label_1.shape, label_0_ndarray.shape
train_label_0 = pd.DataFrame(label_0_ndarray)

train_label_1 = pd.DataFrame(label_1)
train_label_0['label'] = 0

train_label_1['label'] = 1
data_columns = ['env_' + str(i) for i in range(32)] + ['atom_' + str(i) for i in range(32)] + ['label']

train = pd.concat((train_label_0, train_label_1), axis=0)

train.columns = data_columns
train['label'].value_counts()
from keras.layers import Dense, Input

from keras.layers import concatenate

from keras.models import Model

from keras.optimizers import Adam, Adagrad
atom_inp = Input(shape=(32, ))

env_inp = Input(shape=(32, ))

h_1_atom = Dense(100, activation='relu')(atom_inp)

h_1_env = Dense(100, activation='relu')(env_inp)

h_1_combine = concatenate([h_1_atom, h_1_env])

h_2 = Dense(100, activation='relu')(h_1_combine)

out = Dense(1, activation='sigmoid')(h_2)

model = Model(inputs=[atom_inp, env_inp], outputs=out)
model.summary()
X_all = train.drop(columns='label')

y_all = train['label']

X_all._is_copy = False

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
env_columns = ['env_' + str(i) for i in range(32)]

atom_columns = ['atom_' + str(i) for i in range(32)]
X_train[env_columns].head(10)
all_atoms = []

atom_hashs = set()

for atom in atoms:

    a_hash = atom_hash(atom)

    if not atom_hashs.__contains__(a_hash):

        all_atoms.append(atom)

        atom_hashs.add(a_hash)
atom_hash_test = []

for atom in atom_test:

    atom_hash_test.append(atom_hash(atom))
test_env_hash = [atom_hash(env) for env in env_test]
env_test_df = pd.DataFrame(env_test)

env_test_df.columns = ['env_' + str(i) for i in range(32)]

env_test_df['true_atom_hash'] = atom_hash_test

env_test_df['env_hash'] = test_env_hash
all_atom_hash = [atom_hash(atom) for atom in all_atoms]
all_atom_df = pd.DataFrame(all_atoms)

all_atom_df.columns = ['atom_' + str(i) for i in range(32)]

all_atom_df['atom_hash'] = all_atom_hash
env_test_df['tmp'] = 1

all_atom_df['tmp'] = 1

test_data = pd.merge(env_test_df, all_atom_df, how='left', on='tmp')
# opt = Adam(lr=1e-3, decay=1e-3 / 200)

opt = Adagrad(lr=0.001, epsilon=None, decay=0.0)

model.compile(loss="mean_squared_error", optimizer=opt, metrics=['accuracy'])

# model.compile(loss="binary_crossentropy", optimizer=opt,  metrics=['accuracy'])

 

# train the model

print("[INFO] training model...")

model.fit([X_train[env_columns], X_train[atom_columns]], y_train,

          validation_data=([X_val[env_columns], X_val[atom_columns]], y_val),

          epochs=10, batch_size=128, verbose=2)
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
test_data.sample(3)
test_data.columns.__len__()
# make predictions on the testing data

print("[INFO] predicting molecule properties...")

label_predict = model.predict([test_data[env_columns], test_data[atom_columns]])
test_data['predict'] = label_predict
top_k = 1

count_match = 0

for env_hash in test_data['env_hash'].unique():

    a_env = test_data[test_data.env_hash == env_hash]

    true_atom_hash = a_env['true_atom_hash'].values[0]

    a_env_data = a_env[['predict', 'atom_hash']].values

    a_env_data = [(x0, x1) for x0, x1 in a_env_data]

    a_env_data.sort(key=lambda x: x[0], reverse=True)

    top_k_atoms = [x[1] for x in a_env_data[:top_k]]

    if top_k_atoms.__contains__(true_atom_hash):

        count_match += 1
count_match/test_data['env_hash'].unique().__len__()