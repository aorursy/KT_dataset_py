import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
det_descr = pd.read_csv("../input/detectors.csv")
det_descr.set_index(['volume_id', 'layer_id', 'module_id'], inplace=True)

#hits  = pd.read_csv("../input/train_1/event000001160-hits.csv")
#cells = pd.read_csv("../input/train_1/event000001160-cells.csv")
hits  = pd.read_csv("../input/train_1/event000001100-hits.csv")
cells = pd.read_csv("../input/train_1/event000001100-cells.csv")

# Fetch module descriptions for each of the hits
hit_descrs = det_descr.loc[[tuple(x) for x in hits[['volume_id', 'layer_id', 'module_id']].values]]
hits_with_descr = pd.concat([hits, hit_descrs.set_index(hits.index)], axis=1)
hits_with_descr.set_index('hit_id', inplace=True)
del hit_descrs; gc.collect()

# Append hit and module info to each of the cells entries
hits_aug = pd.concat([hits_with_descr.loc[cells.hit_id.values].set_index(cells.index), cells], axis=1)
# Define columns for shorter formulas later
x, y, z = hits_aug.x, hits_aug.y, hits_aug.z

cell_iv = hits_aug.ch1
cell_iu = hits_aug.ch0
cx, cy, cz = hits_aug.cx, hits_aug.cy, hits_aug.cz
pitch_u, pitch_v = hits_aug.pitch_u, hits_aug.pitch_v
module_hv = hits_aug.module_hv
module_hu = hits_aug.module_maxhu

rot_xu, rot_xv, rot_xw = hits_aug.rot_xu, hits_aug.rot_xv, hits_aug.rot_xw
rot_yu, rot_yv, rot_yw = hits_aug.rot_yu, hits_aug.rot_yv, hits_aug.rot_yw
rot_zu, rot_zv, rot_zw = hits_aug.rot_zu, hits_aug.rot_zv, hits_aug.rot_zw

# Calculate the number of cells across each dimension
nu = (module_hu * 2 / pitch_u).round()
nv = (module_hv * 2 / pitch_v).round()

# Some checks for the cell indexes
assert (cell_iv >= 0).all()
assert (cell_iv < nv).all()
assert (cell_iu >= 0).all()
assert (cell_iu < nu).all()

# Calculating locall cell coordinates
hit_v = -module_hv + (cell_iv + 0.5) * pitch_v
hit_u = -module_hu + (cell_iu + 0.5) * pitch_u

# Transforming to global (w = 0, i.e. we're interested in the depth center of the cell)
hit_x = rot_xu * hit_u + rot_xv * hit_v + cx
hit_y = rot_yu * hit_u + rot_yv * hit_v + cy
hit_z = rot_zu * hit_u + rot_zv * hit_v + cz

# Distance between the calculated cell coordinates and the provided hit coordinates
dist_x = hit_x - x
dist_y = hit_y - y
dist_z = hit_z - z

dist_df = pd.DataFrame({'dist_x' : dist_x         ,
                        'dist_y' : dist_y         ,
                        'dist_z' : dist_z         ,
                        'value'  : hits_aug.value ,
                        'hit_id' : hits_aug.hit_id})
dist_df['dist_x_times_value'] = dist_df.dist_x * dist_df.value
dist_df['dist_y_times_value'] = dist_df.dist_y * dist_df.value
dist_df['dist_z_times_value'] = dist_df.dist_z * dist_df.value
g = dist_df.groupby('hit_id')

mean_dist = ((g.dist_x_times_value.sum() / g.value.sum())**2 + \
             (g.dist_y_times_value.sum() / g.value.sum())**2 + \
             (g.dist_z_times_value.sum() / g.value.sum())**2)**0.5
plt.hist(mean_dist, log=True, bins=100);
plt.show()
# Let's also calculate local hit coordinates and compare those with the provided ones
# These are the provided coordinates:
u = (x - cx) * rot_xu + (y - cy) * rot_yu + (z - cz) * rot_zu
v = (x - cx) * rot_xv + (y - cy) * rot_yv + (z - cz) * rot_zv
w = (x - cx) * rot_xw + (y - cy) * rot_yw + (z - cz) * rot_zw

local_df = pd.DataFrame({'u' : u,
                         'v' : v,
                         'w' : w,
                         'cell_u_times_value' : hit_u * hits_aug.value,
                         'cell_v_times_value' : hit_v * hits_aug.value,
                         'value' : hits_aug.value,
                         'hit_id' : hits_aug.hit_id,
                         'tr' : (hits_aug.module_maxhu != hits_aug.module_minhu)})

# And here are the coords calculated from the cell coords:
g2 = local_df.groupby('hit_id')

hit_u = g2.cell_u_times_value.sum() / g2.value.sum()
hit_v = g2.cell_v_times_value.sum() / g2.value.sum()

# Mean over same values to reduce the df
u = g2.u.mean()
v = g2.v.mean()
w = g2.w.mean()
tr = g2.tr.mean().astype(bool)

fig = plt.figure(figsize=(18,5))
ax = fig.add_subplot(141, projection='3d')
ax.scatter((u - hit_u).loc[tr],
           (v - hit_v).loc[tr],
           w          .loc[tr], c='b', marker='o')
ax.scatter((u - hit_u).loc[~tr],
           (v - hit_v).loc[~tr],
           w          .loc[~tr], c='r', marker='^')
ax.set_xlabel('U')
ax.set_ylabel('V')
ax.set_zlabel('W')

ax = fig.add_subplot(142)
ax.scatter((u - hit_u).loc[tr],
           (v - hit_v).loc[tr], c='b', marker='o')
ax.scatter((u - hit_u).loc[~tr],
           (v - hit_v).loc[~tr], c='r', marker='^')
ax.set_xlabel('U')
ax.set_ylabel('V')

ax = fig.add_subplot(143)
ax.scatter((u - hit_u).loc[tr],
           w          .loc[tr], c='b', marker='o')
ax.scatter((u - hit_u).loc[~tr],
           w          .loc[~tr], c='r', marker='^')
ax.set_xlabel('U')
ax.set_ylabel('W')

ax = fig.add_subplot(144)
ax.scatter((v - hit_v).loc[tr],
           w          .loc[tr], c='b', marker='o')
ax.scatter((v - hit_v).loc[~tr],
           w          .loc[~tr], c='r', marker='^')
ax.set_xlabel('V')
ax.set_ylabel('W')

plt.tight_layout()
plt.show()
# let's check the rotations are orthogonal:
rotations = [[hits_aug.rot_xu, hits_aug.rot_xv, hits_aug.rot_xw],
             [hits_aug.rot_yu, hits_aug.rot_yv, hits_aug.rot_yw],
             [hits_aug.rot_zu, hits_aug.rot_zv, hits_aug.rot_zw],]

sum2_x = rotations[0][0]**2 + rotations[0][1]**2 + rotations[0][2]**2
sum2_y = rotations[1][0]**2 + rotations[1][1]**2 + rotations[1][2]**2
sum2_z = rotations[2][0]**2 + rotations[2][1]**2 + rotations[2][2]**2

sum_xy = rotations[0][0]*rotations[1][0] + rotations[0][1]*rotations[1][1] + rotations[0][2]*rotations[1][2]
sum_yz = rotations[1][0]*rotations[2][0] + rotations[1][1]*rotations[2][1] + rotations[1][2]*rotations[2][2]
sum_zx = rotations[2][0]*rotations[0][0] + rotations[2][1]*rotations[0][1] + rotations[2][2]*rotations[0][2]

plt.figure(figsize=(18,8))
plt.subplot(231)
plt.hist(sum2_x, log=True, bins=100);
plt.subplot(232)
plt.hist(sum2_y, log=True, bins=100);
plt.subplot(233)
plt.hist(sum2_z, log=True, bins=100);
plt.subplot(234)
plt.hist(sum_xy, log=True, bins=100);
plt.subplot(235)
plt.hist(sum_yz, log=True, bins=100);
plt.subplot(236)
plt.hist(sum_zx, log=True, bins=100);
# Norm to module plane in local coords is u = 0, v = 0, w = 1.
# Hence in global coordinates it is (rot_xw, rot_yw, rot_zw)
# Let's plot its dot product with two vectors: (cx, cy, 0) and (0, 0, cz):

norm_xy = (det_descr.cx**2 + det_descr.cy**2)**0.5
sel1 = det_descr.module_minhu != det_descr.module_maxhu
sel2 = det_descr.module_minhu == det_descr.module_maxhu

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.scatter(((det_descr.rot_xw * det_descr.cx + det_descr.rot_yw * det_descr.cy) / norm_xy).loc[sel1],
             (det_descr.rot_zw * np.sign(det_descr.cz)).loc[sel1]);
plt.subplot(122)
plt.scatter(((det_descr.rot_xw * det_descr.cx + det_descr.rot_yw * det_descr.cy) / norm_xy).loc[sel2],
             (det_descr.rot_zw * np.sign(det_descr.cz)).loc[sel2]);
