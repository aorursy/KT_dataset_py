# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Let's load some libraries that we'll need to take a look at our data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import multiprocessing as mp

import time

import matplotlib.pyplot as plt

import matplotlib as mpl

import random
##Setup some matplotlib details

font = {'family' : 'serif',

        'weight' : 'normal',

        'size'   : 16}



mpl.rc('font', **font)
#Input the data

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

structures=pd.read_csv('../input/structures.csv')

train.head()

train.head()
structures.head()
#We're giong to start separating our data by type sets: 1JHN and 3JHC

train1JHN=train[train['type']=='1JHN']

test1JHN=test[test['type']=='1JHN']

train3JHC=train[train['type']=='3JHC']

test3JHC=test[test['type']=='3JHC']

print('There are {} unique atoms'.format(structures['atom'].nunique()))

print('Those atoms are {}'.format(structures['atom'].unique()))
def element_to_atomic_number(ele):

    """ Given an input string that is an element's symbol, output its atomic number"""

    ele_to_atomic_number={'C':6, 'H':1, 'N':7, 'O':8, 'F':9}

    return ele_to_atomic_number[ele]

print('The largest value for X is {} and the smallest value is {}'.format(structures['x'].max(),structures['x'].min()))

print('The largest value for Y is {} and the smallest value is {}'.format(structures['y'].max(),structures['y'].min()))

print('The largest value for Z is {} and the smallest value is {}'.format(structures['z'].max(),structures['z'].min()))

#These numbers are important if we end up using conv or capsule nets for regression on the potential

xspread=structures['x'].max()-structures['x'].min()

yspread=structures['y'].max()-structures['y'].min()

zspread=structures['z'].max()-structures['z'].min()

print('The smallest box that would hold all molecules in the data set is: ({0:.2f}x{1:.2f}x{2:.2f}) Angstroms'.format(xspread,yspread,zspread))
#We'll build a list of the molecule names in each type we want to explore

train1JHN.describe()

mol1JHN=train1JHN['molecule_name'].unique()

mol_test1JHN=test1JHN['molecule_name'].unique()

mol3JHC=train3JHC['molecule_name'].unique()

mol_test3JHC=test3JHC['molecule_name'].unique()

structures1JHN=structures[structures['molecule_name'].isin(mol1JHN)]

structures_test1JHN=structures[structures['molecule_name'].isin(mol_test1JHN)]

structures3JHC=structures[structures['molecule_name'].isin(mol3JHC)]

structures_test3JHC=structures[structures['molecule_name'].isin(mol_test3JHC)]
def get_info(struct):

    """get_info(struct) takes in a pandas dataframe object from structures.csv of all structures with a given molecular name 

        and outputs the number of atoms, the total # of electrons, and the molecule's spread in X,Y,Z,"""

    n_atoms=len(struct)

    spread_x=struct['x'].max()-struct['x'].min()

    spread_y=struct['y'].max()-struct['y'].min()

    spread_z=struct['z'].max()-struct['z'].min()

    spread=np.array([spread_x,spread_y,spread_z])

    electrons=0

    for i in range(n_atoms):

        electrons +=element_to_atomic_number(struct.iloc[i]['atom'])

    return n_atoms, electrons,spread

        
sample=5000

n_atoms1JHN=np.zeros((sample,1)).astype(int)

total_electrons1JHN=np.zeros((sample,1)).astype(int)

n_atoms3JHC=np.zeros((sample,1)).astype(int)

total_electrons3JHC=np.zeros((sample,1)).astype(int)

spread1JHN=np.zeros((sample,3))

spread3JHC=np.zeros((sample,3))

for i, name in enumerate(np.random.choice(mol1JHN,sample,replace=False)):

    struct=structures1JHN[structures1JHN['molecule_name']==name]

    n_atoms1JHN[i],total_electrons1JHN[i],spread1JHN[i,:]=get_info(struct)

    if ((i+1)%1000 ==0) and (i !=0):

        print('Currently on the {}th name out of {} train1JHN samples'.format(i, sample))

for i, name in enumerate(np.random.choice(mol3JHC,sample,replace=False)):        

    struct=structures3JHC[structures3JHC['molecule_name']==name]

    n_atoms3JHC[i],total_electrons3JHC[i],spread3JHC[i,:]=get_info(struct)

    if ((i+1)%1000 ==0) and (i !=0):

        print('Currently on the {}th name out of {} train3JHC samples'.format(i, sample))

n_atoms_test1JHN=np.zeros((sample,1)).astype(int)

total_electrons_test1JHN=np.zeros((sample,1)).astype(int)

n_atoms_test3JHC=np.zeros((sample,1)).astype(int)

total_electrons_test3JHC=np.zeros((sample,1)).astype(int)

spread_test1JHN=np.zeros((sample,3))

spread_test3JHC=np.zeros((sample,3))



for i, name in enumerate(np.random.choice(mol_test1JHN,sample,replace=False)):

    struct=structures_test1JHN[structures_test1JHN['molecule_name']==name]

    n_atoms_test1JHN[i],total_electrons_test1JHN[i],spread_test1JHN[i,:]=get_info(struct)

    if ((i+1)%1000 ==0) and (i !=0):

        print('Currently on the {}th name out of {} test1JHN samples'.format(i, sample))



for i, name in enumerate(np.random.choice(mol_test3JHC,sample,replace=False)):        

    struct=structures_test3JHC[structures_test3JHC['molecule_name']==name]

    n_atoms_test3JHC[i],total_electrons_test3JHC[i],spread_test3JHC[i,:]=get_info(struct)

    if ((i+1)%1000 ==0) and (i !=0):

        print('Currently on the {}th name out of {} test3JHC samples'.format(i, sample))
f=plt.figure()

f.set_figheight(5)

f.set_figwidth(12.5)

ax1=plt.subplot(1,2,1)

tmp=plt.hist(n_atoms1JHN,density=True,bins=(n_atoms1JHN.max()-n_atoms1JHN.min()),alpha=1,label='1JHN Training Data')

tmp=plt.hist(n_atoms_test1JHN,density=True, bins=(n_atoms_test1JHN.max()-n_atoms_test1JHN.min()),alpha=.5, color='green',label='1JHN Test Data')

plt.legend(loc='upper left')

plt.xlabel('Number of Atoms in Molecule')

plt.ylabel('Frequency')

plt.title('# of Atoms in 1JHN Molecules')



#Plot # of atoms for 3JHC

ax1=plt.subplot(1,2,2)

tmp=plt.hist(n_atoms3JHC,density=True,bins=(n_atoms3JHC.max()-n_atoms3JHC.min()),alpha=1,label='3JHC Training Data')

tmp=plt.hist(n_atoms_test3JHC,density=True, bins=(n_atoms_test3JHC.max()-n_atoms_test3JHC.min()),alpha=.5, color='green',label='3JHC Test Data')

plt.legend(loc='upper left')

plt.xlabel('Number of Atoms in Molecule')

plt.ylabel('Frequency')

plt.title('# of Atoms in 3JHC Molecules')

plt.show()
g=plt.figure()

g.set_figheight(5)

g.set_figwidth(12.5)



#Plot Z or # of electrons for 1JHN

ax1=plt.subplot(1,2,1)

tmp=plt.hist(total_electrons1JHN,density=True,bins=(total_electrons1JHN.max()-total_electrons1JHN.min()),alpha=1,label='1JHN Training Data')

tmp=plt.hist(total_electrons_test1JHN,density=True, bins=(total_electrons_test1JHN.max()-total_electrons_test1JHN.min()),alpha=.5, color='green',label='1JHN Test Data')

plt.legend(loc='upper left')

plt.xlabel('Number of Electrons in Molecule')

plt.ylabel('Frequency')

plt.title('# of Electrons in 1JHN Molecules')



#Plot Z or # of electrons for 3JHC

ax2=plt.subplot(1,2,2)

tmp=plt.hist(total_electrons3JHC,density=True,bins=(total_electrons3JHC.max()-total_electrons3JHC.min()),alpha=1,label='3JHC Training Data')

tmp=plt.hist(total_electrons_test3JHC,density=True, bins=(total_electrons_test3JHC.max()-total_electrons_test3JHC.min()),alpha=.5, color='green',label='3JHC Test Data')

plt.legend(loc='upper left')

plt.xlabel('Number of Electrons in Molecule')

plt.ylabel('Frequency')

plt.title('# of Electrons in 3JHC molecules')

plt.show()

f=plt.figure()

f.set_figheight(5)

f.set_figwidth(20)

ax1=plt.subplot(1,3,1)

h=plt.hist(np.vstack((spread1JHN[:,0],spread_test1JHN[:,0])).T, density=True, bins=30,label=['X Spread Train1JHN','X Spread Test1JHN'])

plt.legend(loc='upper left')

plt.xlabel('Spread in X')

plt.ylabel('Frequency')

ax1.text(6,0.3,'<X> = {0:.2f} train \nand {1:.2f} test'.format(spread1JHN[:,0].mean(), spread_test1JHN[:,0].mean()))

ax1.text(6,0.23,'<$\sigma_X$> = {0:.2f} train\n and {1:.2f} test'.format(spread1JHN[:,0].std(), spread_test1JHN[:,0].std()))

ax2=plt.subplot(1,3,2)

h=plt.hist(np.vstack((spread1JHN[:,1],spread_test1JHN[:,1])).T, density=True, bins=30,label=['Y Spread Train1JHN','Y Spread Test1JHN'])

plt.legend(loc='upper left')

plt.xlabel('Spread in Y')

plt.ylabel('Frequency')

ax2.text(6,0.3,'<Y> = {0:.2f} train\n and {1:.2f} test'.format(spread1JHN[:,1].mean(), spread_test1JHN[:,1].mean()))

ax2.text(6,0.23,'<$\sigma_Y$> = {0:.2f} train\n and {1:.2f} test'.format(spread1JHN[:,1].std(), spread_test1JHN[:,1].std()))

ax3=plt.subplot(1,3,3)

h=plt.hist(np.vstack((spread1JHN[:,2],spread_test1JHN[:,2])).T, density=True,bins=30,label=['Z Spread Train1JHN','Z Spread Test1JHN'])

plt.legend(loc='upper left')

plt.xlabel('Spread in Z')

plt.ylabel('Frequency')

ax3.text(6,0.3,'<Z> = {0:.2f} train\n and {1:.2f} test'.format(spread1JHN[:,2].mean(), spread_test1JHN[:,2].mean()))

ax3.text(5.5,0.23,'<$\sigma_Z$> = {0:.2f} train\n and {1:.2f} test'.format(spread1JHN[:,2].std(), spread_test1JHN[:,2].std()))

plt.show()
f=plt.figure()

f.set_figheight(5)

f.set_figwidth(20)

ax1=plt.subplot(1,3,1)

h=plt.hist(np.vstack((spread3JHC[:,0],spread_test3JHC[:,0])).T, density=True, bins=30,label=['X Spread Train3JHC','X Spread Test3JHC'])

plt.legend(loc='upper left')

plt.xlabel('Spread in X')

plt.ylabel('Frequency')

ax1.text(6,0.3,'<X> = {0:.2f} train \nand {1:.2f} test'.format(spread3JHC[:,0].mean(), spread_test3JHC[:,0].mean()))

ax1.text(6,0.23,'<$\sigma_X$> = {0:.2f} train\n and {1:.2f} test'.format(spread3JHC[:,0].std(), spread_test3JHC[:,0].std()))

ax2=plt.subplot(1,3,2)

h=plt.hist(np.vstack((spread3JHC[:,1],spread_test3JHC[:,1])).T, density=True, bins=30,label=['Y Spread Train3JHC','Y Spread Test3JHC'])

plt.legend(loc='upper left')

plt.xlabel('Spread in Y')

plt.ylabel('Frequency')

ax2.text(6,0.3,'<Y> = {0:.2f} train\n and {1:.2f} test'.format(spread3JHC[:,1].mean(), spread_test3JHC[:,1].mean()))

ax2.text(6,0.23,'<$\sigma_Y$> = {0:.2f} train\n and {1:.2f} test'.format(spread3JHC[:,1].std(), spread_test3JHC[:,1].std()))

ax3=plt.subplot(1,3,3)

h=plt.hist(np.vstack((spread3JHC[:,2],spread_test3JHC[:,2])).T, density=True,bins=30,label=['Z Spread Train3JHC','Z Spread Test3JHC'])

plt.legend(loc='upper left')

plt.xlabel('Spread in Z')

plt.ylabel('Frequency')

ax3.text(6,0.3,'<Z> = {0:.2f} train\n and {1:.2f} test'.format(spread3JHC[:,2].mean(), spread_test3JHC[:,2].mean()))

ax3.text(5.5,0.23,'<$\sigma_Z$> = {0:.2f} train\n and {1:.2f} test'.format(spread3JHC[:,2].std(), spread_test3JHC[:,2].std()))

plt.show()