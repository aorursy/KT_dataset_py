!pip install FlowIO==0.9.3
%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

plt.rcParams["figure.figsize"] = (15, 10)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

from itertools import cycle

prop_cycle = plt.rcParams['axes.prop_cycle']

colors = prop_cycle.by_key()['color']
from pathlib import Path

import numpy as np

import pandas as pd

import doctest

import copy

from skimage.io import imread

import flowio



# tests help notebooks stay managable

def autotest(func):

    globs = copy.copy(globals())

    globs.update({func.__name__: func})

    doctest.run_docstring_examples(

        func, globs, verbose=True, name=func.__name__)

    return func



# extra arguments accepted for backwards-compatibility (with the fcm-0.9.1 package)

def loadFCS(filename, *args, **kwargs):

    f = flowio.FlowData(filename)

    events = np.reshape(f.events, (-1, f.channel_count))

    channels = []

    for i in range(1, f.channel_count+1):

        key = str(i)

        if 'PnS' in f.channels[key] and f.channels[key]['PnS'] != u' ':

            channels.append(f.channels[key]['PnS'])

        elif 'PnN' in f.channels[key] and f.channels[key]['PnN'] != u' ':

            channels.append(f.channels[key]['PnN'])

        else:

            channels.append('None')

    return FcmData(events, channels)



class FcmData(object):

    def __init__(self, events, channels):

        self.channels = channels

        self.events = events

        self.shape = events.shape



    def __array__(self):

        return self.events
nk_data = Path('..') / 'input' / 'nk_cell_dataset'
nk_data_df = pd.DataFrame({'path': list(nk_data.glob('*/*.fcs'))})

nk_data_df['gated'] = nk_data_df['path'].map(lambda x: x.parent.stem)

nk_data_df.sample(3)
# look at the measured markers

first_fcs = loadFCS(str(nk_data_df['path'].iloc[0]), transform=None, auto_comp=False)

print(first_fcs.channels)
fig, ax1 = plt.subplots(1, 1, figsize=(12, 3))

ax1.imshow(first_fcs.events.T)

ax1.set_aspect(1000)
from sklearn.preprocessing import RobustScaler

chan_norm = RobustScaler()

fig, ax1 = plt.subplots(1, 1, figsize=(30, 30))

ax1.imshow(chan_norm.fit_transform(first_fcs.events).T, cmap='RdBu', vmin=-2, vmax=2)

ax1.set_aspect(1000)

ax1.set_title('Normalized Events')

ax1.set_yticks(range(len(first_fcs.channels)))

ax1.set_yticklabels(first_fcs.channels);
fig, ax1 = plt.subplots(1, 1, figsize=(20, 15))

ax1.plot(chan_norm.fit_transform(first_fcs.events))

ax1.legend(first_fcs.channels)
!ls -lR ../input | grep csv