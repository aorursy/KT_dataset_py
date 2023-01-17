import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

import_path = "../input/"
detectors_df = pd.read_csv(import_path + 'detectors.csv')
detectors_df.head()
train_1_files = os.listdir(import_path + 'train_1')
# Get files names by blocks of 4: cells/hits/particles/truth
sorted(train_1_files)[:4]
cells_df = pd.read_csv(import_path + 'train_1/' + 'event000001000-{}.csv'.format('cells'))
hits_df = pd.read_csv(import_path + 'train_1/' + 'event000001000-{}.csv'.format('hits'))
particles_df = pd.read_csv(import_path + 'train_1/' + 'event000001000-{}.csv'.format('particles'))
truth_df = pd.read_csv(import_path + 'train_1/' + 'event000001000-{}.csv'.format('truth'))
cells_df.head()
particles_df.head()
truth_df.head()
hits_df.head()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

sub_table = hits_df[hits_df['volume_id']==8]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Layer Id')
for i in np.unique(sub_table['layer_id']):
    ax.scatter( sub_table[sub_table['layer_id']==i]['x'],\
                sub_table[sub_table['layer_id']==i]['y'],\
                sub_table[sub_table['layer_id']==i]['z'], 'o', edgecolor='k', label='Layer id: ' + str(i))
ax.view_init(elev=45)
plt.legend()
plt.show()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

sub_table = hits_df[hits_df['volume_id']==8]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Layer Id')
for i in np.unique(sub_table['layer_id']):
    ax.scatter( sub_table[sub_table['layer_id']==i]['x'],\
                sub_table[sub_table['layer_id']==i]['y'],\
                sub_table[sub_table['layer_id']==i]['z'], 'o', edgecolor='k', label='Layer id: ' + str(i))
ax.view_init(elev=85)
plt.legend()
plt.show()
sub_table = hits_df[hits_df['volume_id']==8]

fig = plt.figure(figsize=(15, 15))

n = len(np.unique(sub_table['layer_id']))
for i, layer_id_ in enumerate(np.unique(sub_table['layer_id'])):
    ax = fig.add_subplot( 221 + i, projection='3d')
    ax.set_title('Layer Id:'+str(layer_id_))
    ax.scatter( sub_table[sub_table['layer_id']==layer_id_]['x'],\
                sub_table[sub_table['layer_id']==layer_id_]['y'],\
                sub_table[sub_table['layer_id']==layer_id_]['z'], 'o', edgecolor='k', label='Layer id: ' + str(layer_id_))
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.view_init(elev=60)
    plt.legend()
plt.show()
sub_table = hits_df[hits_df['volume_id']==8]

fig = plt.figure(figsize=(15, 15))

n = len(np.unique(sub_table['layer_id']))
for i, layer_id_ in enumerate(np.unique(sub_table['layer_id'])):
    ax = fig.add_subplot( 221 + i, projection='3d')
    ax.set_title('Layer Id:'+str(layer_id_))
    ax.scatter( sub_table[sub_table['layer_id']==layer_id_]['x'],\
                sub_table[sub_table['layer_id']==layer_id_]['y'],\
                sub_table[sub_table['layer_id']==layer_id_]['z'], 'o', edgecolor='k', label='Layer id: ' + str(layer_id_))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.view_init(elev=90)
    plt.legend()
plt.show()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

sub_table = hits_df[hits_df['volume_id']==8]
layer_ids = np.unique(sub_table['layer_id'])

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

for i, layer_id_ in enumerate(layer_ids):
    ax.scatter(sub_table[sub_table['layer_id']==layer_id_]['x'],\
               sub_table[sub_table['layer_id']==layer_id_]['layer_id'],\
                sub_table[sub_table['layer_id']==layer_id_]['z'], 'o', edgecolor='k', label='Layer id: ' + str(i))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=45)
plt.title('Projection (x, z)')
plt.legend()
plt.show()