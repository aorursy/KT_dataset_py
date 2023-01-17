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



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec # Alignments



import seaborn as sns # theme & dataset

print(f"Matplotlib Version : {mpl.__version__}")

print(f"Seaborn Version : {sns.__version__}")



Summary = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

Item = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')

Info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv')

Station = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')
plt.rcParams['figure.dpi'] = 200
dpi=200
fig, axes = plt.subplots(2, 3, figsize=(8, 5))

plt.show()
fig, axes = plt.subplots(2, 3, figsize=(8, 5))

plt.tight_layout()

plt.show()
# With Subplot2grid



fig = plt.figure(figsize=(8,5))  

#Figure Initialized



ax = [None for _ in range(6)]

#this will make a list to save many ax for setting parameter in each one.



ax[0] = plt.subplot2grid((3,4), (0,0), colspan=4)

ax[1] = plt.subplot2grid((3,4), (1,0), colspan=1)

ax[2] = plt.subplot2grid((3,4), (1,1), colspan=1)

ax[3] = plt.subplot2grid((3,4), (1,2), colspan=1)

ax[4] = plt.subplot2grid((3,4), (1,3), colspan=1,rowspan=2)

ax[5] = plt.subplot2grid((3,4), (2,0), colspan=3)



for ix in range(6): 

    ax[ix].set_title('ax[{}]'.format(ix)) # make ax title for distinguish:)

    ax[ix].set_xticks([]) # to remove x ticks

    ax[ix].set_yticks([]) # to remove y ticks

    

fig.tight_layout()

plt.show()

fig = plt.figure(figsize=(8, 5))



ax = [None for _ in range(3)]





ax[0] = fig.add_axes([0.1,0.1,0.8,0.4]) # x, y, dx, dy

ax[1] = fig.add_axes([0.15,0.6,0.25,0.6])

ax[2] = fig.add_axes([0.5,0.6,0.4,0.3])



for ix in range(3):

    ax[ix].set_title('ax[{}]'.format(ix))

    ax[ix].set_xticks([])

    ax[ix].set_yticks([])



plt.show()
fig = plt.figure(figsize=(8, 5))



gs = fig.add_gridspec(3, 3) # make 3 by 3 grid (row, col)



ax = [None for _ in range(5)]



ax[0] = fig.add_subplot(gs[0, :]) 

ax[0].set_title('gs[0, :]')



ax[1] = fig.add_subplot(gs[1, :-1])

ax[1].set_title('gs[1, :-1]')



ax[2] = fig.add_subplot(gs[1:, -1])

ax[2].set_title('gs[1:, -1]')



ax[3] = fig.add_subplot(gs[-1, 0])

ax[3].set_title('gs[-1, 0]')



ax[4] = fig.add_subplot(gs[-1, -2])

ax[4].set_title('gs[-1, -2]')



for ix in range(5):

    ax[ix].set_xticks([])

    ax[ix].set_yticks([])



plt.tight_layout()

plt.show()
from collections import OrderedDict

cmaps = OrderedDict()

cmaps['Diverging'] = [

            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',

            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',

                        'Dark2', 'Set1', 'Set2', 'Set3',

                        'tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Perceptually Uniform Sequential'] = [

            'viridis', 'plasma', 'inferno', 'magma', 'cividis']



cmaps['Sequential'] = [

            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',

            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',

            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps.items())

gradient = np.linspace(0, 1, 256)

gradient = np.vstack((gradient, gradient))





def plot_color_gradients(cmap_category, cmap_list, nrows):

    fig, axes = plt.subplots(nrows=nrows)

    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)

    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)



    for ax, name in zip(axes, cmap_list):

        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))

        pos = list(ax.get_position().bounds)

        x_text = pos[0] - 0.01

        y_text = pos[1] + pos[3]/2.

        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)



    # Turn off *all* ticks & spines, not just the ones with colormaps.

    for ax in axes:

        ax.set_axis_off()





for cmap_category, cmap_list in cmaps.items():

    plot_color_gradients(cmap_category, cmap_list, nrows)



plt.show()