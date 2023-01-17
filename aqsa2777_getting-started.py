import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib.pyplot import *

import os

import matplotlib.pyplot as plt

from os import listdir

from os.path import isfile, join

print(os.listdir("../input"))

import os

root = '../input/'
csv_files = [f for f in listdir(root) if isfile(join(root, f)) and '.csv' in f ]

dataframes = {}

for file in csv_files:

    dataframes[os.path.splitext(file)[0]] = pd.read_csv(root+file)

for i,file in enumerate(csv_files,1):

    print(i,'file name      :',file)

    print('  dataframe shape:',dataframes[os.path.splitext(file)[0]].shape)

    print('  data columns   : ',end='')

    print(*dataframes[os.path.splitext(file)[0]].columns ,sep=', ',end='\n\n')
train = pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

sub=pd.read_csv('../input/sample_submission.csv')
structures = pd.read_csv('../input/structures.csv')

train.head()
test.head()
print(f'There are {train.shape[0]} rows in train data.')

print(f'There are {test.shape[0]} rows in test data.')



print(f"There are {train['molecule_name'].nunique()} distinct molecules in train data.")

print(f"There are {test['molecule_name'].nunique()} distinct molecules in test data.")

print(f"There are {train['atom_index_0'].nunique()} unique atoms.")

print(f"There are {train['type'].nunique()} unique types.")
train.head()
train.columns

structures.atom.unique()
train.type.unique()
from sklearn import linear_model

feautures=['atom_index_0','atom_index_1']
y = train.scalar_coupling_constant

X = train[feautures]
lm = linear_model.LinearRegression()

model = lm.fit(X,y)
predictions = lm.predict(X)

lm.score(X,y)