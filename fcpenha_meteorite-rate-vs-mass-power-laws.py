# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input/data"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Basic libraries

import numpy as np

import pandas as pd

from scipy import stats



# File related

import zipfile

from subprocess import check_output



# Regression and machine learning

import tensorflow as tf



# Plotting with matplotlib

import matplotlib

import matplotlib.pyplot as plt



from pandas.tools.plotting import parallel_coordinates

from pandas.tools.plotting import andrews_curves

from pandas.tools.plotting import radviz



plt.style.use('fivethirtyeight')

plt.rcParams['axes.labelsize'] = 20

plt.rcParams['axes.titlesize'] = 20

plt.rcParams['xtick.labelsize'] = 18

plt.rcParams['ytick.labelsize'] = 18

plt.rcParams['legend.fontsize'] = 14
from subprocess import check_output

print(check_output(['ls', '../input/']).decode('utf8'))
meteorites = pd.read_csv('../input/meteorite-landings.csv')



meteorites.head()
rate_dict = {}



# Binning

mass_bins_edges = np.logspace(-1,

                              np.log10(meteorites['mass'].max()),

                              num=30, base=10.

                             )



for j in range(len(mass_bins_edges)-1):

    

    mass_bin_center = (mass_bins_edges[j+1] + mass_bins_edges[j]) / 2.

        

    rate_dict[mass_bin_center] = meteorites['mass'][

        (meteorites['mass'] >= mass_bins_edges[j]) &

        (meteorites['mass'] < mass_bins_edges[j+1])

        ].count()

    

rate = pd.Series(rate_dict)



rate
fig, axes = plt.subplots(figsize=(5.,6.))



plt.errorbar(

                rate.index.values,

                rate.values,

                yerr=np.sqrt(rate.values),

                fmt='o',

                ecolor='grey',

                capthick=0.5

                )



axes.set_xscale("log", nonposx='clip')

axes.set_yscale("log", nonposy='clip')



# Plot on the same scale as the plot shown in the introduction

axes.set_xlim([1.e-2, 1.e16])

axes.set_ylim([1.e-2, 2.e4])



axes.set_xlabel(r'$m$ (kg)')

axes.set_ylabel(r'$N$')



axes.set_title(r'Vertical bars are statistical errors: $\sigma=\sqrt{N}$', fontsize=14)



plt.show()

plt.close()
fig, axes = plt.subplots(ncols=2, sharex=True,  figsize=(10.,6.))



axes[0].errorbar(

                rate.index.values,

                rate.values * (rate.index.values ** 0.6),

                yerr=np.sqrt(rate.values) * (rate.index.values ** 0.6),

                fmt='o',

                ecolor='grey',

                capthick=0.5

                )



axes[0].set_xscale("log", nonposx='clip')

axes[0].set_yscale("log", nonposy='clip')



axes[0].set_xlabel(r'$m$ (kg)')

axes[0].set_ylabel(r'$N\times m^p$ (kg${}^{p}$)')



axes[0].set_title(r'$p=0.6$')



axes[1].errorbar(

                rate.index.values,

                rate.values * (rate.index.values ** (-0.6)),

                yerr=np.sqrt(rate.values) * (rate.index.values ** (-0.6)),

                fmt='o',

                ecolor='grey',

                capthick=0.5

                )



axes[1].set_xscale("log", nonposx='clip')

axes[1].set_yscale("log", nonposy='clip')



axes[1].set_xlabel(r'$m$ (kg)')



axes[1].set_title(r'$p=-0.6$')



plt.show()

plt.close()