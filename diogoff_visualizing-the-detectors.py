import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import Axes3D for 3D visualization
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('../input/detectors.csv')

df[['volume_id', 'layer_id', 'module_id', 'cx', 'cy', 'cz']].head()
x_min, x_max = df['cx'].min(), df['cx'].max()
y_min, y_max = df['cy'].min(), df['cy'].max()
z_min, z_max = df['cz'].min(), df['cz'].max()

print('x: %10.2f %10.2f' % (x_min, x_max))
print('y: %10.2f %10.2f' % (y_min, y_max))
print('z: %10.2f %10.2f' % (z_min, z_max))
df['xyz'] = df[['cx', 'cy', 'cz']].values.tolist()

df[['volume_id', 'layer_id', 'module_id', 'xyz']].head()
groupby = df.groupby('volume_id')['xyz'].apply(list).to_frame()

groupby
fig = plt.figure(figsize=(15, 15))

for k in range(groupby.shape[0]):
    ax = fig.add_subplot(3, 3, k+1, projection='3d')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    for (idx, row) in groupby.iloc[:k+1].iterrows():
        xyz = np.array(row['xyz'])
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        ax.plot(x, y, z, linewidth=0.5)
        ax.text(x[0], y[0], z[0], str(idx), None)

plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.show()
groupby = df.groupby(['volume_id', 'layer_id'])['xyz'].apply(list).to_frame()

groupby
fig = plt.figure(figsize=(15, 80))

for k in range(groupby.shape[0]):
    ax = fig.add_subplot(16, 3, k+1, projection='3d')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    for (idx, row) in groupby.iloc[:k+1].iterrows():
        xyz = np.array(row['xyz'])
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        ax.plot(x, y, z, linewidth=0.5)
    ax.text(x[0], y[0], z[0], str(idx), None)

plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.show()