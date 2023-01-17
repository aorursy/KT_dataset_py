from skimage.io import imread # for reading images

import matplotlib.pyplot as plt # for showing plots

from skimage.measure import label # for labeling regions

from skimage.measure import regionprops # for shape analysis

import numpy as np # for matrix operations and array support

from skimage.color import label2rgb # for making overlay plots

import matplotlib.patches as mpatches # for showing rectangles and annotations

from skimage.util.montage import montage2d # for making 3d montages from 2D images
# simple test image diagonal

test_img=np.eye(4)

print('Input Image')

print(test_img)



test_label_4=label(test_img,connectivity=1)

print('Labels with 4-neighborhood')

print(test_label_4)



test_label_8=label(test_img,connectivity=2)

print('Labels with 8-neighborhood')

print(test_label_8)
test_img=np.array([1 if x in [0,13,26] else 0 for x in range(27)]).reshape((3,3,3))

print('Input Image')

print(test_img)



test_label_1=label(test_img,connectivity=1)

print('Labels with Face-sharing')

print(test_label_1)



test_label_2=label(test_img,connectivity=2)

print('Labels with Edge-Sharing')

print(test_label_2)



test_label_3=label(test_img,connectivity=3)

print('Labels with Vertex-Sharing')

print(test_label_3)
em_image_vol = imread('../input/training.tif')

em_thresh_vol = imread('../input/training_groundtruth.tif')

print("Data Loaded, Dimensions", em_image_vol.shape,'->',em_thresh_vol.shape)
em_idx = np.random.permutation(range(em_image_vol.shape[0]))[0]

em_slice = em_image_vol[em_idx]

em_thresh = em_thresh_vol[em_idx]

print("Slice Loaded, Dimensions", em_slice.shape)
# show the slice and threshold

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (9, 4))

ax1.imshow(em_slice, cmap = 'gray')

ax1.axis('off')

ax1.set_title('Image')

ax2.imshow(em_thresh, cmap = 'gray')

ax2.axis('off')

ax2.set_title('Segmentation')

# here we mark the threshold on the original image



ax3.imshow(label2rgb(em_thresh,em_slice, bg_label=0))

ax3.axis('off')

ax3.set_title('Overlayed')
# make connected component labels

em_label = label(em_thresh)



# show the segmentation, labels and overlay

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (9, 4))

ax1.imshow(em_thresh, cmap = 'gray')

ax1.axis('off')

ax1.set_title('Segmentation')

ax2.imshow(em_label, cmap = plt.cm.gist_earth)

ax2.axis('off')

ax2.set_title('Labeling')

# here we mark the threshold on the original image



ax3.imshow(label2rgb(em_label,em_slice, bg_label=0))

ax3.axis('off')

ax3.set_title('Overlayed')
shape_analysis_list = regionprops(em_label)

first_region = shape_analysis_list[0]

print('List of region properties for',len(shape_analysis_list), 'regions')

print('Features Calculated:',', '.join([f for f in dir(first_region) if not f.startswith('_')]))
fig, ax = plt.subplots(figsize=(10, 6))

ax.imshow(label2rgb(em_label,em_slice, bg_label=0))



for region in shape_analysis_list:

    # draw rectangle using the bounding box

    minr, minc, maxr, maxc = region.bbox

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,

                              fill=False, edgecolor='red', linewidth=2)

    ax.add_patch(rect)



ax.set_axis_off()

plt.tight_layout()
fig, ax = plt.subplots(figsize=(10, 6))

ax.imshow(label2rgb(em_label,em_slice, bg_label=0))



for region in shape_analysis_list:

    x1=region.major_axis_length

    x2=region.minor_axis_length

    anisotropy = (x1-x2)/(x1+x2)

    # for anisotropic shapes use red for the others use blue

    print('Label:',region.label,'Anisotropy %2.2f' % anisotropy)

    if anisotropy>0.1:

        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,

                              fill=False, edgecolor='red', linewidth=2)

        ax.add_patch(rect)

    else:

        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,

                              fill=False, edgecolor='green', linewidth=2)

        ax.add_patch(rect)



ax.set_axis_off()

plt.tight_layout()
em_label_vol = label(em_thresh_vol[::2,::2,::2])
%matplotlib inline

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.max(em_label_vol,i).squeeze(), interpolation='none', cmap = plt.cm.gist_earth)

    cax.set_title('%s Projection' % clabel)

    cax.set_xlabel(clabel[0])

    cax.set_ylabel(clabel[1])
%matplotlib inline

print('Labels',np.unique(em_label_vol))

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):

    cax.imshow(np.max(em_label_vol,i).squeeze(), interpolation='none', cmap = plt.cm.gist_earth)

    cax.set_title('%s Projection' % clabel)

    cax.set_xlabel(clabel[0])

    cax.set_ylabel(clabel[1])
# show the segmentation, labels and overlay

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 5))

# downsample the images since they take up a lot of memory (for a kaggle kernel)

label_montage=montage2d(em_label_vol[::3]).astype(int)

slice_montage=montage2d(em_image_vol[::6,::2,::2]/255)



overlay_montage=label2rgb(label_montage,slice_montage,bg_label=0)



ax1.imshow(slice_montage, cmap = 'gray')

ax1.axis('off')

ax1.set_title('Segmentation')

ax2.imshow(label_montage, cmap = plt.cm.gist_earth)

ax2.axis('off')

ax2.set_title('Labeling')

# here we mark the threshold on the original image

ax3.imshow(overlay_montage)

ax3.axis('off')

ax3.set_title('Overlayed')
from sklearn.decomposition import PCA

xx,yy,zz=np.meshgrid(

    range(em_label_vol.shape[1]),

    range(em_label_vol.shape[0]),

    range(em_label_vol.shape[2])

)
def calc_shape_tensor(em_label_vol,idx, show_plot=False):

    shape_pca=PCA(3)

    pixel_pos=np.stack([xx[em_label_vol==idx],yy[em_label_vol==idx],zz[em_label_vol==idx]],-1)

    shape_pca.fit(pixel_pos)

    if show_plot:

        new_pix=shape_pca.transform(pixel_pos)

        fig,(m_axs,p_axs)=plt.subplots(2,3,figsize=(11,8))

        for c_ax,n_ax,(x_dim,y_dim) in zip(m_axs,p_axs,[(1,2),(0,1),(0,2)]):

            x_data=pixel_pos[:,x_dim]-shape_pca.mean_[x_dim]

            y_data=pixel_pos[:,y_dim]-shape_pca.mean_[y_dim]

            #n_std=5*(x_data.std()+y_data.std())

            n_std=1 # keep the pixels in standard coordinates

            c_ax.plot(x_data/n_std,

                  y_data/n_std,'r+')

            c_ax.set_title('Standard Dimension')

            n_ax.plot(new_pix[:,x_dim],

                  new_pix[:,y_dim],'r+')

            n_ax.set_title('Transformed Dimension')

            for i,(c_comp,r_score,c_col) in enumerate(zip(shape_pca.components_.T,

                                          shape_pca.singular_values_,

                                          ['blue','green','black'])):

                c_score=np.sqrt(r_score)/5 # fix this score

                c_ax.plot(c_score * c_comp[x_dim], c_score * c_comp[y_dim], linewidth=2, color=c_col)

                c_ax.quiver(0,0,c_score*c_comp[x_dim],c_score*c_comp[y_dim],

                    zorder=11, width=0.02, scale=6,

                    label='PCA {}'.format(i),color=c_col)

            c_ax.legend()

            c_ax.axis('equal')

            n_ax.axis('equal')

        

    return shape_pca
calc_shape_tensor(em_label_vol,2,True)
calc_shape_tensor(em_label_vol,10,True)
calc_shape_tensor(em_label_vol,20,True)