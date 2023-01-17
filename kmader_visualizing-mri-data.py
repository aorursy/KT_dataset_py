%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from cv2 import imread, createCLAHE # read and equalize images
from glob import glob
import matplotlib.pyplot as plt
import h5py
data_path = os.path.join('..', 'input', 'mri-heart-processing', 'train_mri_128_128.h5')
# show what is inside
with h5py.File(data_path, 'r') as h5_data:
    for c_key in h5_data.keys():
        print(c_key, h5_data[c_key].shape, h5_data[c_key].dtype)
    cur_images = h5_data['image'][0:10]
import matplotlib.animation as animation
Writer = animation.writers['imagemagick']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
from tqdm import tqdm
for ind in tqdm(range(cur_images.shape[0])):
    ims = []
    temp_stack = cur_images[ind,:,:,:]
    plt.close('all')
    fig, ax1 = plt.subplots(1,1, figsize = (8, 8))
    c_aximg = ax1.imshow(temp_stack[0], cmap='bone', interpolation='lanczos', animated = True)
    ax1.axis('off')
    plt.tight_layout()
    def update_image(frame):
        c_aximg.set_array(temp_stack[frame])
        return c_aximg,
    im_ani = animation.FuncAnimation(fig, update_image, 
                                     frames = range(temp_stack.shape[0]),
                                     interval=50, repeat_delay=300,
                                    blit=True)
    im_ani.save('hr_%03d.gif' % (ind), writer=writer)
from skimage.segmentation import slic
def tslic(tstack, numSegments = 200):
    return slic(tstack, 
                n_segments = numSegments,
                compactness = 5e-4,  
                spacing = (1,1,0.1), 
                enforce_connectivity = True,  
                multichannel = False,
                sigma = (0.5,0.5,1))
n_stck = tslic(temp_stack)
plt.close('all')
fig, ax1 = plt.subplots(1,1, figsize = (8, 8))
c_aximg = ax1.imshow(np.sum(n_stck==10, 0), cmap='bone_r', interpolation='lanczos', animated = True)
ax1.axis('off')
plt.tight_layout()
def update_image(frame):
    c_aximg.set_array(n_stck[frame]==10)
    return c_aximg,
im_ani = animation.FuncAnimation(fig, update_image, 
                                 frames = range(n_stck.shape[0]),
                                 interval=50, repeat_delay=300,
                                blit=True)
im_ani.save('slic_%03d.gif' % (ind), writer=writer)
from skimage.viewer import ImageViewer
iv = ImageViewer(n_stck[0])
iv.show()
