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
import pickle



DATA_DIR = '/kaggle/input/data-molecules/'

STRUCTURE_FILE = 'structures.csv'

TEST_FEATURES_FILE = 'test_features.csv'

TRAIN_FEATURES_FILE = 'train_features.csv'



molecules_train_edge = {}

structure_file = pd.read_csv(DATA_DIR + STRUCTURE_FILE)



for train_features_file in pd.read_csv(DATA_DIR + TRAIN_FEATURES_FILE, chunksize=100000):

    for name, data in train_features_file.groupby('molecule_name'):

        num_nodes_molecule = structure_file.loc[structure_file['molecule_name'] == name].shape[0]

        molecule = np.zeros(( num_nodes_molecule ** 2, 9))

        data_ = data.values

        for row in data_:

            i = row[7]

            j = row[8]

            m = row[[18, 19, 20, 21, 23, 24, 25, 26, 27]]

            molecule[i * num_nodes_molecule + j] = m

            molecule[i + j * num_nodes_molecule] = m

      

        molecules_train_edge[name] = molecule  

    print(len(molecules_train_edge)) 

    with open('train_edge_features.pkl', 'wb') as f:

        pickle.dump(molecules_train_edge, f)

    with open('train_edge_features.pkl', 'rb') as f:

        molecules_train_edge = pickle.load(f)

    print(len(molecules_train_edge))

    

print('finish_gen')
df = pd.read_csv(DATA_DIR + TRAIN_FEATURES_FILE)

print(df['molecule_name'].nunique())

print('_________________________')

print('DONE')
from subprocess import check_output

print(check_output(["ls", "../working"]).decode("utf8"))
from IPython.display import FileLink

FileLink(r'train_edge_features.pkl')