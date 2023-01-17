# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import_path = "../input/"

# Any results you write to the current directory are saved as output.
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
import io
import base64
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import cm
# https://www.kaggle.com/drgilermo/dynamics-of-new-york-city-animation/code
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

N = 200

bounds = []
axes_labels = ['x', 'y', 'z']
for label_ in axes_labels:
    bounds += [np.min(hits_df[label_][:N]), np.max(hits_df[label_][:N])]

def animate(hits_id):
    ax.clear()
    ax.set_title('LHC Particles Hits')    
    ax.scatter(hits_df['x'][:hits_id], hits_df['y'][:hits_id], hits_df['z'][:hits_id], s=np.ones(hits_id)*50, c='red', marker='o', edgecolor='black', alpha=0.5)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_zlim(bounds[4], bounds[5])
    
ani = animation.FuncAnimation(fig,animate, np.arange(0, N), interval = 5)
plt.close()
ani.save('animation.gif', writer='imagemagick', fps=2)
filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))