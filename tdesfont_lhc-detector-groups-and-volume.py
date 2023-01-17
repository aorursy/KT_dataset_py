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
for feature in ['volume_id', 'layer_id', 'module_id']:
    print(feature, ':', np.unique(hits_df[feature]))
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

yticks = np.unique(hits_df['volume_id'])

colors = [(round(np.random.random(),2), round(np.random.random(),2), round(np.random.random(),2)) for i in yticks]

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')
for i, volume_id_ in enumerate(yticks):
    ax.plot(hits_df[hits_df['volume_id'] == volume_id_]['x'],\
            hits_df[hits_df['volume_id'] == volume_id_]['volume_id'],\
            hits_df[hits_df['volume_id'] == volume_id_]['z'],\
            'o', alpha=0.2, color=colors[i], label='Volume_id:{}'.format(volume_id_))
ax.set_xlabel('x')
ax.set_ylabel('Y Volume_id')
ax.set_zlabel('z')
ax.view_init(azim=45, elev=45)
plt.legend()
plt.show()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

yticks = np.unique(hits_df['volume_id'])

colors = [(round(np.random.random(),2), round(np.random.random(),2), round(np.random.random(),2)) for i in yticks]

fig = plt.figure(figsize=(15,15))

# subplot grids
grid_index = list(range(331, 331+9))
azimuts = [0, 45, 90]
elev = [0, 45, 90]

for grid_i in range(0, 3):
    for grid_j in range(0, 3):
        
        ax = fig.add_subplot(grid_index[grid_i*3 + grid_j], projection='3d')
        for i, volume_id_ in enumerate(yticks):
            ax.plot(hits_df[hits_df['volume_id'] == volume_id_]['x'],\
                    hits_df[hits_df['volume_id'] == volume_id_]['volume_id'],\
                    hits_df[hits_df['volume_id'] == volume_id_]['z'],\
                    'o', alpha=0.2, color=colors[i], label='Volume_id:{}'.format(volume_id_))
        ax.set_xlabel('x')
        ax.set_ylabel('Y Volume_id')
        ax.set_zlabel('z')
        ax.view_init(azim=azimuts[grid_i], elev=elev[grid_j])
    
fig.tight_layout()
plt.legend()
plt.show()