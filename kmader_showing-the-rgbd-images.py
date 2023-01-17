from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import matplotlib.pyplot as plt

import matplotlib as mpl

import numpy as np

from glob import glob

import os

import skimage.filters as flts

from skimage.io import imread # read in images
all_pairs = set(map(lambda x: os.path.splitext(x)[0],glob('../input/3d_scenes/*.*')))

img_dict_fcn = lambda base_path: (base_path,{ext: imread('%s.%s' % (base_path,ext)) for ext in ('tiff','png')})

print(all_pairs)
all_images = list(map(img_dict_fcn,all_pairs))
%matplotlib inline

img_dict = all_images[0][1]

train_data = img_dict['tiff']

lab_data = img_dict['png']

fig, all_maxs = plt.subplots(len(all_images),2, figsize = (5,20))

for maxs, (_, img_dict) in zip(all_maxs,all_images):

    for ax, (ext, img_data) in zip(maxs,img_dict.items()):

        ax.imshow(img_data)

        ax.axis('off')

        ax.set_title(ext)
raw_shape = all_images[0][1]['png'].shape

xx,yy = np.meshgrid(range(raw_shape[1]),range(raw_shape[0]))

# flip xx

xx = xx[:,::-1]



mvertices = np.array([xx[::8,::8].flatten(),

                      yy[::8,::8].flatten(),

                      all_images[0][1]['png'][::8,::8].flatten()])

print(xx.shape,yy.shape,raw_shape,mvertices.shape)
%matplotlib inline

from mpl_toolkits.mplot3d.axes3d import *

import matplotlib.pyplot as plt

from matplotlib import cm

fig = plt.figure(figsize = (15,15))

ax = Axes3D(fig)

xx,yy = np.meshgrid(range(raw_shape[1]),range(raw_shape[0]))

# flip xx

xx = xx[:,::-1]

Zdata = all_images[0][1]['png'][::8,::8].astype(np.float32)

Zdata -= Zdata.mean()

Zdata /= -1*Zdata.std()

ax.plot_surface(xx[::8,::8],yy[::8,::8],Zdata, rstride=1, cstride=1, 

                facecolors=all_images[0][1]['tiff'][::8,::8].astype(np.float32)/255,

                linewidth=0, antialiased=True)

ax.set_zlim3d((-1,1))

ax.view_init(70,90)
len(all_images)
BLOCK_SIZE = 128

N_CUT = 16

out_blocks = []

for _,img_dict in all_images:

    for i in range(1,11):

        train_data = img_dict['tiff'][::i,::i]

        lab_data = img_dict['png'][::i,::i]

        print(lab_data.shape)

        def get_rnd_bounds():

            return (np.random.randint(train_data.shape[0]-BLOCK_SIZE),

                    np.random.randint(train_data.shape[1]-BLOCK_SIZE))

        def get_block(in_img,bnds):

            return in_img[bnds[0]:(bnds[0]+BLOCK_SIZE),bnds[1]:(bnds[1]+BLOCK_SIZE)]

        out_blocks += [(get_block(train_data,rwnd),get_block(lab_data,rwnd)) for rwnd in 

                       map(lambda x: get_rnd_bounds(),range(50))]

rgb_blk_stack = np.stack([a for a,b in out_blocks])[:,:,:,:3]

bw_blk_stack = np.stack([b for a,b in out_blocks])

print(rgb_blk_stack.shape,bw_blk_stack.shape)
rgb_train = rgb_blk_stack.swapaxes(1,3).swapaxes(2,3)

bw_segs = np.expand_dims(bw_blk_stack,1).astype(np.float32)

bw_segs -= bw_segs[:,:,N_CUT:BLOCK_SIZE-N_CUT,N_CUT:BLOCK_SIZE-N_CUT].mean()

bw_segs /= np.abs(bw_segs[:,:,N_CUT:BLOCK_SIZE-N_CUT,N_CUT:BLOCK_SIZE-N_CUT]).max()

print(rgb_train.shape,bw_segs.shape)
sc_train, sc_segs = rgb_train, bw_segs

sc_train = sc_train[:,:,N_CUT:BLOCK_SIZE-N_CUT,N_CUT:BLOCK_SIZE-N_CUT]

sc_segs = sc_segs[:,:,N_CUT:BLOCK_SIZE-N_CUT,N_CUT:BLOCK_SIZE-N_CUT]
def show_rgbd_maps(rgb_images, seg_images, count = 3):

    batch_idx = np.array(range(rgb_images.shape[0]))

    np.random.shuffle(batch_idx)

    fig, ax_all = plt.subplots(2, count, figsize = (12,8))

    for pid, (c_raw_ax, c_flt_ax) in zip(batch_idx,ax_all.T):

        c_raw_ax.imshow(rgb_images[pid,:,:,:].swapaxes(0,2).swapaxes(0,1))

        c_raw_ax.set_title("Color Image:{}".format(pid))

        c_raw_ax.axis('off')

        c_flt_ax.imshow(seg_images[pid,0,:,:], cmap='gray', vmin = -1, vmax = 1)

        c_flt_ax.set_title("Depth Image:{}".format(pid))

        c_flt_ax.axis('off')
show_rgbd_maps(sc_train, sc_segs,6)
np.savez_compressed('depth_training_data.npz', rgb_images = sc_train, depth_maps = sc_segs)