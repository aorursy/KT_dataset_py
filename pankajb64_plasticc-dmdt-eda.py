# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
import multiprocessing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tnrange, tqdm_notebook
from collections import OrderedDict
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime, timedelta
from matplotlib import gridspec

print(os.listdir("../input/"))
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/PLAsTiCC-2018/training_set.csv')
df_meta = pd.read_csv('../input/PLAsTiCC-2018/training_set_metadata.csv')
df['unix_time'] = (df['mjd'] - 40587)*86400
df['datetime'] = pd.to_datetime(df['unix_time'], unit='s')
objects = df_meta['object_id'].values
def load_dmdt_images(objects, base_dir='train'):
    dmdt_img_dict = OrderedDict()
    for obj in objects:
        key = '{}/{}_dmdt.pkl'.format(base_dir, obj)
        if os.path.isfile(key):
            with(open(key, 'rb')) as f:
                dmdt_img_dict[obj] = pickle.load(f)
    return dmdt_img_dict
dmdt_img_dict = load_dmdt_images(objects, '../input/plasticc_dmdt_images/train/train')
classes = np.sort(df_meta['target'].drop_duplicates().values)
fig = plt.figure(figsize=(15,6))
ax = sns.countplot(df_meta['target'])
samples = df_meta.groupby('target')['object_id', 'target'].head(1).values
def gen_plots(df, samples):
    for sample in samples:
        fig = plt.figure(figsize=(21,9))
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        outer_grid = gridspec.GridSpec(1, 2)
        object_id = sample[0]
        label = sample[1]
        df_obj = df.loc[df.object_id == object_id]
        gen_flux_plots(df_obj, object_id, label, outer_grid[0], fig)
        viz_dmdt(object_id, label, outer_grid[1], fig, cbar_ax)
        fig.suptitle("Time-series Light Curve and DMDT Images for all 6 passband for object ID - {} of class {}".format(object_id, label), fontsize=16)
        #rect=[0, 0, 0.91, 0.95]
        fig.tight_layout(rect=[0, 0, 0.91, 0.95])
def gen_flux_plot(df, ax, labels):
    passband = df['passband'].drop_duplicates().values[0]
    label = labels[passband]
    sns.scatterplot(ax=ax, x=df['datetime'], y=df['flux'], label=label)
    ax.set_xlim(df.iloc[0]['datetime'] - timedelta(days=20), df.iloc[-1]['datetime'] + timedelta(days=20))
def gen_flux_plots(df, object_id, label, outer_grid, fig):
    ax = fig.add_subplot(outer_grid)
    labels = ['u', 'g', 'r', 'i', 'z', 'Y']
    sps = df.groupby('passband').apply(lambda x : gen_flux_plot(x, ax, labels))
    ax.legend()
    fig.add_subplot(ax)
    #fig.suptitle('Time-series Light Curve for all 6 passbands for object - {} of class {}'.format(object_id, label), fontsize=16)
def viz_dmdt(object_id, label, outer_grid, fig, cbar_ax):
    dmdt_img = dmdt_img_dict[object_id]
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_grid)
    shared_ax = None
    for i in range(6): #num passband
        i_idx = 0 if i < 3 else 1
        j_idx = i%3
        gs = inner_grid[i_idx, j_idx]
        ax = fig.add_subplot(gs) if shared_ax is None else fig.add_subplot(gs, sharex=shared_ax, sharey=shared_ax)
        sns.heatmap(ax=ax, data=dmdt_img[:,:,i], cmap="hot", cbar=(i==0), cbar_ax=None if i else cbar_ax)
    #fig.suptitle("DMDT Images for all 6 passband for object ID - {} of class {}".format(object_id, label), fontsize=16)
    
gen_plots(df, samples)
