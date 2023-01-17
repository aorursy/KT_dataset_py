%matplotlib inline
import os
import numpy as np
from skimage.io import imread
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage import filters as skthresh
from skimage.morphology import opening, closing, disk
from scipy.ndimage import binary_fill_holes
from skimage.morphology import label
from skimage.segmentation import mark_boundaries
from tqdm import tqdm_notebook as tqdm
from skimage.feature import greycomatrix, greycoprops
base_dir = os.path.join('..', 'input')
all_tiffs = glob(os.path.join(base_dir, 'nmc*/*/grayscale/*'))
tiff_df = pd.DataFrame(dict(path = all_tiffs))
tiff_df['frame'] = tiff_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
tiff_df['experiment'] = tiff_df['frame'].map(lambda x: '_'.join(x.split('_')[0:-1]))
tiff_df['slice'] = tiff_df['frame'].map(lambda x: int(x.split('_')[-1]))
print('Images Found:', tiff_df.shape[0])
tiff_df.sample(3)
random_path = tiff_df.sample(1, random_state = 123)['path'].values[0]
bw_img = imread(random_path)[250:-250, 250:-250]
# make to 8-bit since texture analysis is much easier with fixed levels
bw_img = np.clip(255.0*bw_img/bw_img.max(), 0, 255).astype(np.uint8)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10), dpi=120)
ax1.imshow(bw_img, cmap='bone')
ax1.set_title('Gray Scale')

thresh_img = bw_img > skthresh.threshold_triangle(bw_img)
ax2.imshow(thresh_img, cmap='bone')
ax2.set_title('Segmentation')
bw_seg_img = opening(
    closing(
        opening(thresh_img, disk(3)),
        disk(1)
    ), disk(3)
)
bw_seg_img = binary_fill_holes(bw_seg_img)
ax3.imshow(bw_seg_img, cmap='bone')
ax3.set_title('Clean Segments');
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 24), dpi=100)
bw_lab_img = label(bw_seg_img)
ax1.imshow(bw_lab_img, cmap = 'nipy_spectral')
# find boundaries
ax3.imshow(mark_boundaries(label_img = bw_lab_img, image = bw_img))
ax3.set_title('Boundaries');
# compute some GLCM properties each patch
grayco_prop_list = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

out_df_list = []
for patch_idx in tqdm(np.unique(bw_lab_img[bw_lab_img>0])):
    xx_box, yy_box = np.where(bw_lab_img==patch_idx)
    gray_roi_img = bw_img[xx_box.min():xx_box.max(), 
                                 yy_box.min():yy_box.max()].copy()
    lab_roi_img = bw_lab_img[xx_box.min():xx_box.max(), 
                                 yy_box.min():yy_box.max()]
    gray_roi_img[lab_roi_img!=patch_idx] = 0
    glcm = greycomatrix(gray_roi_img,
                        [5], [0], 256, symmetric=True, normed=True)
    
    out_row = dict(
        intensity_mean=np.mean(bw_img[bw_lab_img == patch_idx]),
        intensity_std=np.std(bw_img[bw_lab_img == patch_idx]),
        index = patch_idx)
    
    for c_prop in grayco_prop_list:
        out_row[c_prop] = greycoprops(glcm, c_prop)[0, 0]
        
    out_df_list += [out_row]
out_df=pd.DataFrame(out_df_list)
out_df.sample(3)
from sklearn.cluster import KMeans
n_groups = 3
km_grouper = KMeans(n_clusters = n_groups, random_state = 2018)
out_df['group'] = ['G%02d' % (i) for i in km_grouper.fit_predict(out_df.drop(['index'], axis = 1))]
import seaborn as sns
sns.pairplot(out_df.drop(['index'], axis = 1),
             hue='group')
n_samples = 4
fig, m_axs = plt.subplots(n_groups, n_samples, figsize = (20, 20))
for (g_name, c_df), s_axs in zip(out_df.groupby('group'), m_axs):
    for c_ax, (_, c_row) in zip(s_axs, c_df.sample(n_samples).iterrows()):
        xx_box, yy_box = np.where(bw_lab_img==c_row['index'])
        gray_roi_img = bw_img[xx_box.min():xx_box.max(), 
                                 yy_box.min():yy_box.max()]
        lab_roi_img = bw_lab_img[xx_box.min():xx_box.max(), 
                                 yy_box.min():yy_box.max()]
        c_ax.imshow(mark_boundaries(label_img = (lab_roi_img==c_row['index']).astype(int),
                                    image = gray_roi_img))
        c_ax.axis('off')
        c_ax.set_title('Group: {}'.format(g_name))
n_groups = 4
km_grouper = KMeans(n_clusters = n_groups, random_state = 2018)
out_df['group_2'] = ['G%02d' % (i) for i in km_grouper.fit_predict(out_df[grayco_prop_list])]

n_samples = 5
fig, m_axs = plt.subplots(n_groups, n_samples, figsize = (20, 20))
for (g_name, c_df), s_axs in zip(out_df.groupby('group_2'), m_axs):
    for c_ax, (_, c_row) in zip(s_axs, c_df.sample(n_samples).iterrows()):
        xx_box, yy_box = np.where(bw_lab_img==c_row['index'])
        gray_roi_img = bw_img[xx_box.min():xx_box.max(), 
                                 yy_box.min():yy_box.max()]
        lab_roi_img = bw_lab_img[xx_box.min():xx_box.max(), 
                                 yy_box.min():yy_box.max()]
        c_ax.imshow(mark_boundaries(label_img = (lab_roi_img==c_row['index']).astype(int),
                                    image = gray_roi_img))
        c_ax.axis('off')
        c_ax.set_title('Group: {}'.format(g_name))
