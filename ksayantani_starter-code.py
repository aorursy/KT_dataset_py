# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data       , CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



import matplotlib

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as plt

%matplotlib inline



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
structures = pd.read_csv("../input/structures.csv")
molecule = 'dsgdb9nsd_000001'
a = df_train.loc[df_train['molecule_name'] == f'{molecule}']

b = structures[structures.molecule_name == f'{molecule}']
list(df_train['type'].unique())
def convert_index_to_atom(a, b, atom_index):

    c = a.merge(b, how='left', left_on=['molecule_name', f'atom_index_{atom_index}'], right_on=['molecule_name', 'atom_index'])

    c.drop('atom_index', axis=1, inplace=True)

    c.rename(columns = {

    'atom': f'atom_{atom_index}',

    'x': f'x_{atom_index}',

    'y': f'y_{atom_index}',

    'z': f'z_{atom_index}'

}, inplace=True)

    c.drop(f'atom_index_{atom_index}', axis=1, inplace=True)

    return c
c = convert_index_to_atom(df_train, structures, 0)

c = convert_index_to_atom(c, structures, 1)
t = c[c['molecule_name'] == f'{molecule}']
types  = list(c.groupby('type').groups)

types
fig, ax = plt.subplots(figsize=(10, 5))

x = c[c['type'] == '1JHC']['x_0']

y = c[c['type'] == '1JHC']['y_0']

ax.scatter(x, y)

ax.grid(True)

ax.set_title('1JHC')

fig.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(10, 5))

x = c[c['type'] == '1JHN']['x_0']

y = c[c['type'] == '1JHN']['y_0']

ax.scatter(x, y)

ax.grid(True)

ax.set_title('1JHN')

fig.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(10, 5))

x = c[c['type'] == '2JHC']['x_0']

y = c[c['type'] == '2JHC']['y_0']

ax.scatter(x, y)

ax.grid(True)

ax.set_title('2JHC')

fig.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(10, 5))

x = c[c['type'] == '2JHH']['x_0']

y = c[c['type'] == '2JHH']['y_0']

ax.scatter(x, y)

ax.grid(True)

ax.set_title('2JHH')

fig.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(10, 5))

x = c[c['type'] == '2JHN']['x_0']

y = c[c['type'] == '2JHN']['y_0']

ax.scatter(x, y)

ax.grid(True)

ax.set_title('2JHN')

fig.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(10, 5))

x = c[c['type'] == '3JHC']['x_0']

y = c[c['type'] == '3JHC']['y_0']

ax.scatter(x, y)

ax.grid(True)

ax.set_title('3JHC')

fig.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(10, 5))

x = c[c['type'] == '3JHH']['x_0']

y = c[c['type'] == '3JHH']['y_0']

ax.scatter(x, y)

ax.grid(True)

ax.set_title('3JHH')

fig.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(10, 5))

x = c[c['type'] == '3JHN']['x_0']

y = c[c['type'] == '3JHN']['y_0']

ax.scatter(x, y)

ax.grid(True)

ax.set_title('3JHN')

fig.tight_layout()

plt.show()