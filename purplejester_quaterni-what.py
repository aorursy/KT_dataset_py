%matplotlib notebook
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

import pandas as pd

from pyquaternion import Quaternion
x_trn = pd.read_csv('../input/X_train.csv')
ser0 = x_trn[x_trn.series_id == 0]
orient = ser0.columns.str.startswith('orientation')
qs = [Quaternion(list(row)) for _, row in ser0[ser0.columns[orient]].iterrows()]
vec = [1, 0, 1]

xyz = [q.rotate(vec) for q in qs]

xs, ys, zs = [list(seq) for seq in zip(*xyz)]

ax = plt.axes(projection='3d')

ax.scatter3D(xs, ys, zs, c=zs)