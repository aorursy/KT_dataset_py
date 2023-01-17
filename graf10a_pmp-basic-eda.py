import os

import numpy as np

import pandas as pd

from pathlib import Path



print(os.listdir("../input"))
%%time 

path=Path('../input')

train=pd.read_csv(path/'train.csv')

test=pd.read_csv(path/'test.csv')

struct=pd.read_csv(path/'structures.csv')
print(f'The shape of train is {train.shape}')

print(f'The shape of test is {test.shape}')

print(f'The shape of struct is {struct.shape}')

print(f"\nThe number of NA's in train is {train.isna().sum().sum()}.")

print(f"The number of NA's in test is {test.isna().sum().sum()}.")

print(f"The number of NA's in struct is {struct.isna().sum().sum()}.")

print(f"\nThe column names of train are \n{train.columns}.")

print(f"\nThe column names of test are \n{test.columns}.")

print(f"\nThe column names of struct are \n{struct.columns}.")
train.head(20)
test.head(20)
struct.head(20)
%%time



cols=test.columns



data=pd.concat([train[cols], test])
data.describe()
print(f"The number of molecules in the train set is {train['molecule_name'].nunique()}.")

print(f"The number of molecules in the test set is {test['molecule_name'].nunique()}")



print(f"\nThe number of interaction types in the train set is {train['type'].nunique()}.")

print(f"The number of interaction types in the test set is {test['type'].nunique()}")

print(f"The number of interaction types in the train and test sets combined is {data['type'].nunique()}")



print(f"\nThe number of atomic types in struct is {struct['atom'].nunique()}")
types_count = data['type'].value_counts()

types_order = types_count.index.values

types_count
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



width = 22

height = 7

fs = '24'



plt.figure(figsize=(width, height))



sns.set(font_scale=1.6)



plt.subplot(1, 3, 1)

sns.countplot(data['type'], order=types_order)

plt.title('train+test', fontsize=fs)

plt.ylabel('Counts')



plt.subplot(1, 3, 2)

sns.countplot(train['type'], order=types_order)

plt.title('train', fontsize=fs)

plt.ylabel('Counts')



plt.subplot(1, 3, 3)

sns.countplot(test['type'], order=types_order)

plt.title('test', fontsize=fs)

plt.ylabel('Counts')



plt.tight_layout()
struct['atom'].value_counts()
width = 6

height = 6

fs = '20'

fs_label = '17'



plt.figure(figsize=(width, height))



sns.set(font_scale=1.2)



sns.countplot(x='atom', data=struct, order = struct['atom'].value_counts().index)

plt.title("Atoms present in the 'struct' file", fontsize=fs)

plt.xlabel('Atoms', fontsize=fs_label)

plt.ylabel('Counts', fontsize=fs_label)



plt.tight_layout()
print(f"The minimum values of the 'atom_index_1' and 'atom_index_2' are {data.atom_index_0.min()}" \

      f" and {data.atom_index_1.min()}, respectively.")



print(f"The maximum values of the 'atom_index_1' and 'atom_index_2' are {data.atom_index_0.max()}" \

      f" and {data.atom_index_1.max()}, respectively.")
atom_counts = data.groupby('molecule_name').size().reset_index(name='count')

atom_counts.head()
atom_counts.tail()
print(f"The total number of molecules in train and test is {len(atom_counts)}.")

print(f"The minimum number of couplings per molecule is {np.min(atom_counts['count'].values)}.")

print(f"The maximum number of couplings per moluecule is {np.max(atom_counts['count'].values)}.")
coupling_types = train['type'].unique()



coupling_types
%%time



vsize = 4

hsize = 2



plt.figure()

fig, ax = plt.subplots(vsize,hsize,figsize=(18,20))



for (i, ct) in enumerate(coupling_types):

    i += 1

    plt.subplot(vsize, hsize, i)



    sns.distplot(train.loc[train['type'] == ct, 'scalar_coupling_constant'], color='blue', bins=60, label=ct)

    

    plt.title("Scalar Coupling Type "+ct, fontsize='20')

    plt.xlabel('Scalar Coupling Constant', fontsize='16')

    plt.ylabel('Density', fontsize='16')

    locs, labels = plt.xticks()

    plt.tick_params(axis='x', which='major', labelsize=16)#, pad=-40)

    plt.tick_params(axis='y', which='major', labelsize=16)

    #plt.legend(loc='best', fontsize='16')

    

plt.tight_layout()    

plt.show()