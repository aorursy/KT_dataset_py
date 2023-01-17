# Llamado de Librerías básicas

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Paths Kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

np.random.seed(12345)



dict_new = {

    'student': np.random.choice(['pooh', 'rabbit', 'piglet', 'Christopher'], 50000, p=[0.5, 0.1, 0.1, 0.3]),

    'test': np.random.choice(['pooh', 'rabbit', 'piglet', 'Christopher'], 50000, p=[0.25, 0.25, 0.25, 0.25]),

    'user': np.random.choice(['pooh', 'rabbit', 'piglet'], 50000, p=[0.30, 0.35, 0.35])

}



z = pd.DataFrame(dict_new)



z
