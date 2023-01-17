# Import all the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from ase import Atoms

import ase.visualize

from mpl_toolkits.mplot3d import Axes3D



sns.set()
import os

input_folder = '/kaggle/input'

os.listdir(input_folder)
# Obtain all the information possible from the csv files.

input_folder = '/kaggle/input'



potential_energy = pd.read_csv(f'{input_folder}/potential_energy.csv')

dipole_moments = pd.read_csv(f'{input_folder}/dipole_moments.csv')

mulliken_charges = pd.read_csv(f'{input_folder}/mulliken_charges.csv')

magnetic_shielding_tensors = pd.read_csv(f'{input_folder}/magnetic_shielding_tensors.csv')

scalar_coupling_contributions = pd.read_csv(f'{input_folder}/scalar_coupling_contributions.csv')

structure = pd.read_csv(f'{input_folder}/structures.csv')

characteristics = [potential_energy, dipole_moments, mulliken_charges, magnetic_shielding_tensors, scalar_coupling_contributions, structure]

for element in characteristics:

    print(element.info())

    print('')
# Check out how are the data distributed in each dataframe, see the dimensions to know which to concatenate



for element in characteristics:

    print(element.head())

    print(element.shape[0])

    print('')
# Concatenate potential energy and dipole moments



energy_dipole = pd.concat([potential_energy, dipole_moments],axis=1)

energy_dipole = energy_dipole.loc[:,~energy_dipole.columns.duplicated()]

energy_dipole.head()
# Concatenate mulliken charges and the magnetic shielding tensors



mulliken_magnetic = pd.concat([mulliken_charges, magnetic_shielding_tensors],axis=1)

mulliken_magnetic = mulliken_magnetic.loc[:,~mulliken_magnetic.columns.duplicated()]
# Reduce the size of the structure file in order to have the information for all particles



other_items = set(mulliken_magnetic['molecule_name'].value_counts().index)

all_items = set(structure['molecule_name'].value_counts().index)

compliment = all_items.intersection(other_items)

reduced_structure = structure[structure.molecule_name.isin(compliment)]
# Add more variables to the mulliken_magnetic_structure



mulliken_magnetic_structure = mulliken_magnetic

mulliken_magnetic_structure['atom'] = list(reduced_structure['atom'])

mulliken_magnetic_structure['x'] = list(reduced_structure['x'])

mulliken_magnetic_structure['y'] = list(reduced_structure['y'])

mulliken_magnetic_structure['z'] = list(reduced_structure['z'])

# Add the number of atoms pero particle

energy_dipole['number_of_atoms'] = list(mulliken_magnetic_structure.molecule_name.value_counts().sort_index())


# print dataframes:

mulliken_magnetic_structure.head()
energy_dipole.head()
energy_dipole.describe().T
# Show the distributions for each energy and dipole moment



plt.hist(energy_dipole.potential_energy, bins = 100, color='red')

plt.title('Potential Energy Distribution')

plt.xlabel('Potential Energy')

plt.ylabel('Count')

plt.show()
sns.pairplot(energy_dipole[['X','Y','Z']], diag_kind="kde")

plt.suptitle('Dipole moment Components \n')

plt.show()
sns.violinplot(energy_dipole.number_of_atoms, showmeans=True, color='Green')

plt.title('Distribution and dispersion of the number of atoms')

plt.xlabel('Number of atoms')

plt.show()
mms = mulliken_magnetic_structure

molecule = energy_dipole['molecule_name']


# Matplotlib visualization:



fig = plt.figure()

fig = plt.figure(figsize=plt.figaspect(0.5))



for m_id in range(30):

    ax = fig.add_subplot(5,6,m_id+1, projection='3d')

    ax.scatter(mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'H')]['x'],

               mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'H')]['y'],

               mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'H')]['z'],

               c='w', s=100, linewidths=1, edgecolors='k')



    ax.scatter(mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'C')]['x'],

               mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'C')]['y'],

               mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'C')]['z'],

               c='k', s=100, linewidths=1, edgecolors='k')



    ax.scatter(mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'O')]['x'],

               mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'O')]['y'],

               mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'O')]['z'],

               c='r', s=100, linewidths=1, edgecolors='k')



    ax.scatter(mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'N')]['x'],

               mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'N')]['y'],

               mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'N')]['z'],

               c='b', s=100,  linewidths=1, edgecolors='k')



    ax.scatter(mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'F')]['x'],

               mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'F')]['y'],

               mms[(mms.molecule_name == molecule[m_id])&(mms.atom == 'F')]['z'],

               c='orange', s=100,  linewidths=1, edgecolors='k')

    plt.xticks([])

    plt.yticks([])

    plt.grid()

    plt.title(f'{molecule[m_id]}')

plt.subplots_adjust(left=.02, right=1.8, top=4)



plt.show()
mms = mulliken_magnetic_structure

molecule = energy_dipole['molecule_name']



m_id = np.random.randint(0,len(molecule))



symbols = np.array(mms[(mms.molecule_name == molecule[m_id])].atom)

location = np.array(mms[(mms.molecule_name == molecule[m_id])].loc[:,['x','y','z']])

system = Atoms(positions=location, symbols=symbols)

ase.visualize.view(system, viewer="x3d")

#print(system.get_chemical_formula())
av_mulliken_charge = mms.groupby('molecule_name').mean()['mulliken_charge']

plt.hist(av_mulliken_charge, bins=12,color='orange')

plt.title('Average Mulliken Charge distribution')

plt.ylabel('Count')

plt.xlabel('Average Mulliken Charge')

plt.show()
net_mulliken_charge = mms.groupby('molecule_name').sum()['mulliken_charge']

plt.hist(net_mulliken_charge, bins=12,color='orange')

plt.title('Net Mulliken Charge distribution')

plt.ylabel('Count')

plt.xlabel('Net Mulliken Charge')

plt.show()
plt.subplot(121)

plt.scatter(net_mulliken_charge*1000000, energy_dipole.potential_energy,c='g', s=.4)

plt.xlabel('Net Mulliken Charge')

plt.ylabel('Potential Energy')

plt.title('Scatterplot for Net Mulliken Charge and the Potential Energy')



plt.subplot(122)

plt.scatter(av_mulliken_charge*1000000, energy_dipole.potential_energy, c='r', s=.4)

plt.subplots_adjust(left=.01,right=2)

plt.xlabel('Net Mulliken Charge')

plt.ylabel('Potential Energy')

plt.title('Scatterplot for Average Mulliken Charge and the Potential Energy')
scc = scalar_coupling_contributions

scc['scalar_coupling_constant'] = scc['fc']+scc['sd']+scc['pso']+scc['dso']
sns.pairplot(scc[['fc','sd','pso','dso','type']])


types = scc.type.value_counts()

plt.bar(types.index,types)
sns.distplot(scc.scalar_coupling_constant, color='g', kde=False)
plt.figure(figsize=(10,10))

sns.boxplot(x='type', y='scalar_coupling_constant', data=scc)

plt.xlabel('Type')

plt.ylabel('Scalar Coupling Constant')