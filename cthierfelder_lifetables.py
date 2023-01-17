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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
df = pd.read_csv('/kaggle/input/life-table-g20/lifeTable.csv')
df.columns
plt.plot(df['age'],df['Germany_Female'],label='German female')
plt.plot(df['age'],df['Germany_Male']  ,label='German male')
plt.plot(df['age'],df['Germany_Female']-df['Germany_Male']  ,label='gender difference')
plt.xlabel('age')
plt.ylabel('number of survivors')
plt.grid(True)
plt.legend()
def draw_3d(verts, ymin, ymax, line_at_zero=True, colors=True):

    if line_at_zero:
        zeroline = 0
    else:
        zeroline = ymin
    for p in verts:
        p.insert(0, (p[0][0], zeroline))
        p.append((p[-1][0], zeroline))

    if colors:
        hue = 0.9
        huejump = .027
        facecolors = []
        edgecolors = []
        for v in verts:
            hue = (hue - huejump) % 1
            c = mcolors.hsv_to_rgb([hue, 1, 1])
                                    # random.uniform(.8, 1),
                                    # random.uniform(.7, 1)])
            edgecolors.append(c)
            # Make the facecolor translucent:
            facecolors.append(mcolors.to_rgba(c, alpha=.37))
    else:
        facecolors = (1, 1, 1, .8)
        edgecolors = (0, 0, 1, 1)

    poly = PolyCollection(verts,facecolors=facecolors, edgecolors=edgecolors)

    zs = range(len(verts))
    # zs = range(len(data)-1, -1, -1)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')

    plt.tight_layout(pad=2.0, w_pad=10.0, h_pad=3.0)

    ax.add_collection3d(poly, zs=zs, zdir='y')
    
    ax.set_xlabel('Age')
    ax.set_ylabel('countries')
    ax.set_zlabel('number of survivors')

    ax.set_xlim3d(0, 140)
    ax.set_ylim3d(-1, len(verts))
    ax.set_zlim3d(ymin, ymax)
    
draw_3d([df[['age','Russia_Male']].to_numpy().tolist(),
        df[['age','Latvia_Male']].to_numpy().tolist(),
        df[['age','Poland_Male']].to_numpy().tolist(),
        df[['age','Belgium_Male']].to_numpy().tolist(),
        df[['age','Germany_Male']].to_numpy().tolist(),
        df[['age','Japan_Male']].to_numpy().tolist()], 0, 100000, colors=True)
plt.show()

