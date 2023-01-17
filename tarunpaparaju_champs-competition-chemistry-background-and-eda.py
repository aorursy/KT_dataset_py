import os

import gc

import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
os.listdir('../input')
train = pd.read_csv('../input/train.csv')

train.head(10)
typelist = list(train['type'].value_counts().index)

typelist
sns.distplot(train['scalar_coupling_constant'], color='orangered')

plt.show()
plt.figure(figsize=(26, 24))

for i, col in enumerate(typelist):

    plt.subplot(4,2, i + 1)

    sns.distplot(train[train['type']==col]['scalar_coupling_constant'],color ='indigo')

    plt.title(col)
dipole_moments = pd.read_csv('../input/dipole_moments.csv')

dipole_moments.head(10)
sns.distplot(dipole_moments.X, color='mediumseagreen')

plt.title('Dipole moment along X-axis')

plt.show()

sns.distplot(dipole_moments.Y, color='seagreen')

plt.title('Dipole moment along Y-axis')

plt.show()

sns.distplot(dipole_moments.Z, color='green')

plt.title('Dipole moment along Z-axis')

plt.show()
plt.figure(figsize=(26, 24))

for i, col in enumerate(typelist):

    plt.subplot(4,2, i + 1)

    sns.distplot(dipole_moments[train['type']==col]['X'],color = 'orange', kde=False)

    sns.distplot(dipole_moments[train['type']==col]['Y'],color = 'red', kde=False)

    sns.distplot(dipole_moments[train['type']==col]['Z'],color = 'blue', kde=False)

    plt.title(col)
potential_energy = pd.read_csv('../input/potential_energy.csv')

potential_energy.head(10)
sns.distplot(potential_energy.potential_energy, color='darkblue', kde=False)

plt.show()
plt.figure(figsize=(26, 24))

for i, col in enumerate(typelist):

    plt.subplot(4,2, i + 1)

    sns.distplot(potential_energy[train['type']==col]['potential_energy'], color = 'orangered')

    plt.title(col)
magnetic_shielding_tensors = pd.read_csv('../input/magnetic_shielding_tensors.csv')

magnetic_shielding_tensors.head(10)
def is_outlier(points, thresh=3.5):

    """

    Returns a boolean array with True if points are outliers and False 

    otherwise.



    Parameters:

    -----------

        points : An numobservations by numdimensions array of observations

        thresh : The modified z-score to use as a threshold. Observations with

            a modified z-score (based on the median absolute deviation) greater

            than this value will be classified as outliers.



    Returns:

    --------

        mask : A numobservations-length boolean array.



    References:

    ----------

        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and

        Handle Outliers", The ASQC Basic References in Quality Control:

        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 

    """

    if len(points.shape) == 1:

        points = points[:,None]

    median = np.median(points, axis=0)

    diff = np.sum((points - median)**2, axis=-1)

    diff = np.sqrt(diff)

    med_abs_deviation = np.median(diff)



    modified_z_score = 0.6745 * diff / med_abs_deviation



    return modified_z_score > thresh
sns.distplot(magnetic_shielding_tensors.XX[~is_outlier(magnetic_shielding_tensors.XX)], color='red')

plt.title('Magnetic Shielding (XX)')

plt.show()

sns.distplot(magnetic_shielding_tensors.XY[~is_outlier(magnetic_shielding_tensors.XY)], color='orangered')

plt.title('Magnetic Shielding (XY)')

plt.show()

sns.distplot(magnetic_shielding_tensors.XZ[~is_outlier(magnetic_shielding_tensors.XZ)], color='orange')

plt.title('Magnetic Shielding (XZ)')

plt.show()

sns.distplot(magnetic_shielding_tensors.YX[~is_outlier(magnetic_shielding_tensors.YX)], color='yellow')

plt.title('Magnetic Shielding (YX)')

plt.show()

sns.distplot(magnetic_shielding_tensors.YY[~is_outlier(magnetic_shielding_tensors.YY)], color='green')

plt.title('Magnetic Shielding (YY)')

plt.show()

sns.distplot(magnetic_shielding_tensors.YZ[~is_outlier(magnetic_shielding_tensors.YZ)], color='blue')

plt.title('Magnetic Shielding (YZ)')

plt.show()

sns.distplot(magnetic_shielding_tensors.ZX[~is_outlier(magnetic_shielding_tensors.ZX)], color='darkblue')

plt.title('Magnetic Shielding (ZX)')

plt.show()

sns.distplot(magnetic_shielding_tensors.ZY[~is_outlier(magnetic_shielding_tensors.ZY)], color='indigo')

plt.title('Magnetic Shielding (ZY)')

plt.show()

sns.distplot(magnetic_shielding_tensors.ZZ[~is_outlier(magnetic_shielding_tensors.ZZ)], color='darkviolet')

plt.title('Magnetic Shielding (ZZ)')

plt.show()
mulliken_charges = pd.read_csv('../input/mulliken_charges.csv')

mulliken_charges.head(10)
sns.distplot(mulliken_charges.mulliken_charge, color = 'seagreen')

plt.show()
sns.distplot(mulliken_charges.loc[mulliken_charges.atom_index == 0].mulliken_charge, color = 'blue')

plt.title('Atom index 0')

plt.show()

sns.distplot(mulliken_charges.loc[mulliken_charges.atom_index == 1].mulliken_charge, color = 'darkblue')

plt.title('Atom index 1')

plt.show()

sns.distplot(mulliken_charges.loc[mulliken_charges.atom_index == 2].mulliken_charge, color = 'blueviolet')

plt.title('Atom index 2')

plt.show()

sns.distplot(mulliken_charges.loc[mulliken_charges.atom_index == 3].mulliken_charge, color = 'purple')

plt.title('Atom index 3')

plt.show()

sns.distplot(mulliken_charges.loc[mulliken_charges.atom_index == 3].mulliken_charge, color = 'indigo')

plt.title('Atom index 4')

plt.show()