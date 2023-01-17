import sys

sys.path.append("../input")



import h5py

import numpy as np

from voxelgrid import VoxelGrid

import matplotlib.pyplot as plt

from matplotlib import animation
sample_index = 0



with h5py.File('../input/train_small.h5','r') as hf:

    sample = hf[str(sample_index)]

    

    sample_data = (sample['points'][:], sample.attrs['label'])

    print("label:",sample_data[1])
x = y = z = 32



sample_voxelgrid = VoxelGrid(sample_data[0], x_y_z=[x,y,z])
sample3d = sample_voxelgrid.vector.reshape(x,y,z)
fig, ax = plt.subplots();
def animate(i, direction):

    ax.set_title(str(i+1) + ' of ' + str(sample3d.shape[0]))

    ax.set_axis_off()

    if direction == 'x':

        return ax.imshow(sample3d[i, :, :], cmap="gray", origin="lower")

    if direction == 'y':

        return ax.imshow(sample3d[:, i, :], cmap="gray", origin="lower")

    if direction == 'z':

        return ax.imshow(sample3d[:, :, i], cmap="gray", origin="lower")
anim_x = animation.FuncAnimation(fig, animate, frames=np.arange(0,32), fargs='x')

anim_y = animation.FuncAnimation(fig, animate, frames=np.arange(0,32), fargs='y')

anim_z = animation.FuncAnimation(fig, animate, frames=np.arange(0,32), fargs='z')
anim_x.save('anim_x.gif', writer='imagemagick')

anim_y.save('anim_y.gif', writer='imagemagick')

anim_z.save('anim_z.gif', writer='imagemagick')