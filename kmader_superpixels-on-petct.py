import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from skimage.util.montage import montage2d

import os

import h5py
%matplotlib inline

with h5py.File(os.path.join('..', 'input', 'lab_petct_vox_5.00mm.h5'), 'r') as p_data:

    id_list = list(p_data['ct_data'].keys())

    print(list(p_data.keys()))

    ct_image = p_data['ct_data'][id_list[0]].value

    pet_image = p_data['pet_data'][id_list[0]].value

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 4))

ct_proj = np.mean(ct_image,1)[::-1]

suv_max = np.sqrt(np.max(pet_image,1)[::-1])

ax1.imshow(ct_proj, cmap = 'bone')

ax1.set_title('CT Image')

ax2.imshow(suv_max, cmap = 'magma')

ax2.set_title('SUV Image')
from skimage.segmentation import slic

from skimage.segmentation import mark_boundaries

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (10, 4))

n_ct_img = (ct_proj+1024).clip(0,2048)/2048

ct_segs = slic(n_ct_img, n_segments = 100, compactness = 0.1)

ax1.imshow(ct_proj, cmap = 'bone')

ax1.set_title('CT Image')

ax2.imshow(ct_segs, cmap = plt.cm.rainbow)

ax2.set_title('Segmented Image')

ax3.imshow(mark_boundaries(n_ct_img, ct_segs))
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (10, 4))

n_suv_img = (suv_max).clip(0,5)/5.0

pet_segs = slic(n_suv_img, n_segments = 100, compactness = 0.05)

ax1.imshow(suv_max, cmap = 'magma')

ax1.set_title('SUV Image')

ax2.imshow(pet_segs, cmap = plt.cm.rainbow)

ax2.set_title('Segmented Image')

ax3.imshow(mark_boundaries(n_suv_img, pet_segs))
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (14, 6))

petct_segs = slic(np.stack([n_ct_img, n_suv_img],-1), n_segments = 100, compactness = 0.05)



ax1.imshow(suv_max, cmap = 'magma')

ax1.set_title('SUV Image')

ax2.imshow(mark_boundaries(n_ct_img, petct_segs))

ax2.set_title('Segmented CT')

ax3.imshow(mark_boundaries(n_suv_img, petct_segs))

ax3.set_title('Segmented PET')
petct_segs = slic(np.stack([np.stack([(ct_slice+1024).clip(0,2048)/2048, 

                            np.sqrt((suv_slice).clip(0,5)/5.0)

                           ],-1) for ct_slice, suv_slice in zip(ct_image, pet_image)],0), 

                  n_segments = 500, 

                  compactness = 0.1,

                 multichannel = True)



petct_max_segs = np.max(petct_segs,1)[::-1]

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (14, 6))

ax1.imshow(suv_max, cmap = 'magma')

ax1.set_title('SUV Image')

ax2.imshow(petct_max_segs, cmap = plt.cm.rainbow)

ax2.set_title('Segmented Image')

ax3.imshow(mark_boundaries(n_suv_img, petct_max_segs))
bright_segs = np.zeros_like(petct_segs)

kept_comps = 0

for i in np.unique(petct_segs):

    if pet_image[petct_segs == i].mean()>1.5:

        bright_segs[petct_segs == i] = 1

        kept_comps+=1

print('Kept', kept_comps,'of', len(np.unique(petct_segs)))

bright_sum_segs = np.sum(bright_segs,1)[::-1]

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (14, 6))

ax1.imshow(suv_max, cmap = 'magma')

ax1.set_title('SUV Image')

ax2.imshow(bright_sum_segs, cmap = plt.cm.bone)

ax2.set_title('Segments Image')
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

def show_3d_mesh(image, threshold):

    p = image[::-1].swapaxes(1,2)

    

    verts, faces = measure.marching_cubes(p, threshold)



    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.15, edgecolor='none', linewidth = 0.1)

    mesh.set_facecolor([.1, 1, .1])

    mesh.set_edgecolor([1, 0, 0])

    

    ax.add_collection3d(mesh)



    ax.set_xlim(0, p.shape[0])

    ax.set_ylim(0, p.shape[1])

    ax.set_zlim(0, p.shape[2])

    

    ax.view_init(80, 5)

    return fig
_ = show_3d_mesh(bright_segs, 0)
from mpl_toolkits.mplot3d.axes3d import Axes3D

def show_pet_3d(image, pet_signal, threshold):

    p = image[::-1].swapaxes(1,2)



    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(121, projection='3d')

    

    verts, faces = measure.marching_cubes(p, 0)

    mesh = Poly3DCollection(verts[faces], alpha=0.15, edgecolor='none', linewidth = 0.1)

    mesh.set_facecolor([.1, 1, .1])

    mesh.set_edgecolor([1, 0, 0])

    

    ax1.add_collection3d(mesh)



    ax1.set_xlim(0, p.shape[0])

    ax1.set_ylim(0, p.shape[1])

    ax1.set_zlim(0, p.shape[2])

    

    ax1.view_init(80, 5)

    

    ax2 = fig.add_subplot(122, projection='3d')

    p_pet = pet_signal[::-1].swapaxes(1,2)

    

    verts, faces = measure.marching_cubes(p_pet, threshold)

    mesh = Poly3DCollection(verts[faces], alpha=0.15, edgecolor='none', linewidth = 0.1)

    mesh.set_facecolor([1, 0, .1])

    mesh.set_edgecolor([.1, 0, 1.0])

    

    ax2.add_collection3d(mesh)



    ax2.set_xlim(0, p.shape[0])

    ax2.set_ylim(0, p.shape[1])

    ax2.set_zlim(0, p.shape[2])

    ax2.view_init(80, 5)

    return fig
bright_seg_pet = pet_image.copy()

bright_seg_pet[bright_segs==0] = 0

_ = show_pet_3d(bright_segs, bright_seg_pet, 1.5)