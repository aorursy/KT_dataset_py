import numpy as np

import matplotlib.pyplot as plt

import os

import vtk

import scipy.io

#import pyvista as pv

import scipy

import skimage

import ipywidgets as wi

import skimage.filters

plt.ion()
voxel_data_file = scipy.io.loadmat('../input/rat-volume/voxel_data.mat')

ID_classes = voxel_data_file['ID_classes']

vox_mat = voxel_data_file['vox_mat']

x_pos = voxel_data_file['x_pos']

y_pos = voxel_data_file['y_pos']

z_pos = voxel_data_file['z_pos']



surface_data_file = scipy.io.loadmat('../input/rat-volume/surface_data.mat')

surface_data = surface_data_file['surface_data']
print(np.unique(vox_mat))

for i in range(len(ID_classes)):

    idx = ID_classes[i][0][0][0]

    name = ID_classes[i][1][0]

    print("%i %s" % (idx, name))
mask_01 = np.where(vox_mat == 1,1,0)

mask_02 = np.where(vox_mat == 2,1,0)

mask_03 = np.where(vox_mat == 3,1,0)

mask_04 = np.where(vox_mat == 4,1,0)

mask_05 = np.where(vox_mat == 5,1,0)

mask_06 = np.where(vox_mat == 6,1,0)

mask_07 = np.where(vox_mat == 7,1,0)

mask_08 = np.where(vox_mat == 8,1,0)

mask_09 = np.where(vox_mat == 9,1,0)

mask_10 = np.where(vox_mat == 10,1,0)

mask_11 = np.where(vox_mat == 11,1,0)
weighted_mask = mask_11*1.0 + mask_10*0.04 + mask_03*0.03
@wi.interact(z=wi.IntSlider(min=10, max=800, continuous_update=False,value=700), d=wi.IntSlider(min=0, max=50, continuous_update=False,value=16))

def gen_rat_pos(z=700,d=16):

    

    fig = plt.figure(figsize=(12,12))  # create a figure object

    ax = fig.add_subplot(2, 1, 1)  # create an axes object in the figure

    ax.set_aspect('equal')

    proj = np.sum(mask_01,axis=0)

    plt.pcolormesh(proj)

    

    plt.axvline(z,c='r')

    plt.fill_betweenx(np.arange(np.shape(proj)[0]),z-d,z+d,alpha=0.6,color='w')

    



    ax = fig.add_subplot(2, 1, 2)  # create an axes object in the figure

    ax.set_aspect('equal')

    proj = np.sum(mask_01,axis=1)



    plt.pcolormesh(proj)

    plt.axvline(z,c='r')

    plt.fill_betweenx(np.arange(np.shape(proj)[0]),z-d,z+d,alpha=0.6,color='w')

    

    plt.show()
@wi.interact(z=wi.IntSlider(min=600,max=800,value=700,continuous_update=False),d=wi.IntSlider(min=1,max=50,value=16,continuous_update=False),smooth=wi.FloatSlider(min=0,max=20,value=5,continuous_update=False))

def genimg(z=700 ,d=16,smooth=5):

    tmp = mask_11[:,:,z-d:z+d]

    summe = np.sum(tmp,axis=2)

    summe = np.rot90(summe,k=-1)

    summe = skimage.filters.gaussian(summe,sigma=smooth)

    fig = plt.figure(figsize=(12,12))

    ax = fig.add_subplot(1, 1, 1)

    ax.set_aspect('equal')

    ax.pcolormesh(summe,cmap='jet')

    ax.set_xlim([100,220])

    ax.set_ylim([150,300])

    plt.title('Tissue type 11')

    plt.show()



    return None

@wi.interact(z=wi.IntSlider(min=600,max=800,value=700,continuous_update=False),d=wi.IntSlider(min=1,max=50,value=16,continuous_update=False),smooth=wi.FloatSlider(min=0,max=20,value=2,continuous_update=False))

def genimg2(z=700 ,d=16,smooth=5):

    tmp = mask_10[:,:,z-d:z+d]

    summe = np.sum(tmp,axis=2)

    summe = np.rot90(summe,k=-1)

    summe = skimage.filters.gaussian(summe,sigma=smooth)

    fig = plt.figure(figsize=(12,12))

    ax = fig.add_subplot(1, 1, 1)

    ax.set_aspect('equal')

    ax.pcolormesh(summe,cmap='gist_gray')

    ax.set_xlim([100,220])

    ax.set_ylim([150,300])

    plt.title('Tissue type 11')

    plt.show()



    return None

@wi.interact(z=wi.IntSlider(min=600,max=800,value=688,continuous_update=False),d=wi.IntSlider(min=1,max=50,value=16,continuous_update=False),smooth=wi.FloatSlider(min=0,max=20,value=7,continuous_update=False))

def genimg2(z=700 ,d=16,smooth=5):

    tmp = weighted_mask[:,:,z-d:z+d]

    summe = np.sum(tmp,axis=2)

    summe = np.rot90(summe,k=-1)

    summe = skimage.filters.gaussian(summe,sigma=smooth)

    fig = plt.figure(figsize=(12,12))

    ax = fig.add_subplot(1, 1, 1)

    ax.set_aspect('equal')

    ax.pcolormesh(summe,cmap='jet')

    ax.set_xlim([100,220])

    ax.set_ylim([150,300])

    plt.title('Weighted map')

    plt.show()



    return None
