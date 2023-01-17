import numpy as np

import skimage.transform

from scipy import ndimage

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

%matplotlib inline



x, y = np.meshgrid(range(8), range(8))





def blur_img(c_img): return (ndimage.zoom(c_img.astype('float'),

                                          3,

                                          order=3,

                                          prefilter=False)*4).astype(int).clip(1, 4)-1





text_imgs = [blur_img(c_img)

             for c_img in [x % 2,

                           y % 2,

                           (x % 2+y % 2)/2.0,

                           (x % 4+y % 2)/2.5,

                           (x % 4+y % 3)/3.5,

                           ((x+y) % 3)/2.0]]



fig, m_axs = plt.subplots(2, 3, figsize=(20, 10))



for c_ax, c_img in zip(m_axs.flatten(), text_imgs):

    sns.heatmap(c_img, annot=False, fmt='2d', ax=c_ax,

                cmap='viridis', vmin=0, vmax=3)

from skimage.feature.texture import greycomatrix

from skimage.util import montage as montage2d



def montage_nd(in_img):

    if len(in_img.shape) > 3:

        return montage2d(np.stack([montage_nd(x_slice) for x_slice in in_img], 0))

    elif len(in_img.shape) == 3:

        return montage2d(in_img)

    else:

        warn('Input less than 3d image, returning original', RuntimeWarning)

        return in_img





dist_list = np.linspace(1, 6, 15)

angle_list = np.linspace(0, 2*np.pi, 15)





def calc_coomatrix(in_img):

    return greycomatrix(image=in_img,

                        distances=dist_list,

                        angles=angle_list,

                        levels=4)





def coo_tensor_to_df(x): return pd.DataFrame(

    np.stack([x.ravel()]+[c_vec.ravel() for c_vec in np.meshgrid(range(x.shape[0]),

                                                                 range(

                                                                     x.shape[1]),

                                                                 dist_list,

                                                                 angle_list,

                                                                 indexing='xy')], -1),

    columns=['E', 'i', 'j', 'd', 'theta'])





coo_tensor_to_df(calc_coomatrix(text_imgs[0])).head(5)
fig, m_axs = plt.subplots(3, 6, figsize=(20, 10))



for (c_ax, d_ax, f_ax), c_img in zip(m_axs.T, text_imgs):

    c_ax.imshow(c_img, vmin=0, vmax=4, cmap='gray')

    c_ax.set_title('Pattern')

    full_coo_matrix = calc_coomatrix(c_img)

    d_ax.imshow(montage_nd(full_coo_matrix), cmap='gray')

    d_ax.set_title('Co-occurence Matrix\n{}'.format(full_coo_matrix.shape))

    d_ax.axis('off')

    avg_coo_matrix = np.mean(full_coo_matrix*1.0, (0, 1))

    f_ax.imshow(avg_coo_matrix, cmap='gray')

    f_ax.set_title('Average Co-occurence\n{}'.format(avg_coo_matrix.shape))
text_df = coo_tensor_to_df(calc_coomatrix(text_imgs[0]))

text_df['ij_diff'] = text_df.apply(lambda x: x['i']-x['j'], axis=1)



simple_corr_df = text_df.groupby(['ij_diff', 'd', 'theta']).agg({

    'E': 'mean'}).reset_index()

simple_corr_df.head(5)
def grouped_weighted_avg(values, weights, by):

    return (values * weights).groupby(by).sum() / weights.groupby(by).sum()





fig, m_axs = plt.subplots(3, 6, figsize=(20, 10))



for (c_ax, d_ax, f_ax), c_img in zip(m_axs.T, text_imgs):

    c_ax.imshow(c_img, vmin=0, vmax=4, cmap='gray')

    c_ax.set_title('Pattern')

    full_coo_matrix = calc_coomatrix(c_img)

    text_df = coo_tensor_to_df(full_coo_matrix)

    text_df['ij_diff'] = text_df.apply(lambda x: x['i']-x['j'], axis=1)



    simple_corr_df = text_df.groupby(['ij_diff', 'd', 'theta']).agg({

        'E': 'sum'}).reset_index()

    gwa_d = grouped_weighted_avg(

        simple_corr_df.ij_diff, simple_corr_df.E, simple_corr_df.d)

    d_ax.plot(gwa_d.index, gwa_d.values)

    d_ax.set_title('Distance Co-occurence')



    gwa_theta = grouped_weighted_avg(

        simple_corr_df.ij_diff, simple_corr_df.E, simple_corr_df.theta)

    f_ax.plot(gwa_theta.index, gwa_theta.values)

    f_ax.set_title('Angular Co-occurence')
import seaborn as sns

from skimage.color import label2rgb

import numpy as np

import matplotlib.pyplot as plt

import h5py



slice_img = (h5py.File('../input/train/mri_00004519.h5')['image'][100][:, :, 0]/3200*255).astype('uint8')

slice_mask = slice_img>0



# show the slice and threshold

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 5))

ax1.imshow(slice_img, cmap='gray')

ax1.axis('off')

ax1.set_title('Image')

ax2.imshow(slice_mask, cmap='gray')

ax2.axis('off')

ax2.set_title('Segmentation')

# here we mark the threshold on the original image



ax3.imshow(label2rgb(slice_mask > 0, slice_img, bg_label=0, ))

ax3.axis('off')

ax3.set_title('Overlayed')
xx, yy = np.meshgrid(

    np.arange(slice_img.shape[1]),

    np.arange(slice_img.shape[0]))

region_labels = (xx//48) * 64+yy//48

region_labels = region_labels.astype(int)

sns.heatmap(region_labels[::48, ::48].astype(int),

            annot=True,

            fmt="03d",

            cmap='nipy_spectral',

            cbar=False,

            )
# compute some GLCM properties each patch

from skimage.feature import greycomatrix, greycoprops

from tqdm import tqdm_notebook as tqdm

grayco_prop_list = ['contrast', 'dissimilarity',

                    'homogeneity', 'energy',

                    'correlation', 'ASM']



prop_imgs = {}

for c_prop in grayco_prop_list:

    prop_imgs[c_prop] = np.zeros_like(slice_img, dtype=np.float32)

score_img = np.zeros_like(slice_img, dtype=np.float32)

out_df_list = []

for patch_idx in tqdm(np.unique(region_labels)):

    xx_box, yy_box = np.where(region_labels == patch_idx)



    glcm = greycomatrix(slice_img[xx_box.min():xx_box.max(),

                                   yy_box.min():yy_box.max()],

                        [5], [0], 256, symmetric=True, normed=True)



    mean_score = np.mean(slice_mask[region_labels == patch_idx])

    score_img[region_labels == patch_idx] = mean_score



    out_row = dict(

        intensity_mean=np.mean(slice_img[region_labels == patch_idx]),

        intensity_std=np.std(slice_img[region_labels == patch_idx]),

        score=mean_score)



    for c_prop in grayco_prop_list:

        out_row[c_prop] = greycoprops(glcm, c_prop)[0, 0]

        prop_imgs[c_prop][region_labels == patch_idx] = out_row[c_prop]



    out_df_list += [out_row]
# show the slice and threshold

fig, m_axs = plt.subplots(2, 4, figsize=(20, 10))

n_axs = m_axs.flatten()

ax1 = n_axs[0]

ax2 = n_axs[1]

ax1.imshow(slice_img, cmap='gray')

ax1.axis('off')

ax1.set_title('Image')

ax2.imshow(score_img, cmap='gray')

ax2.axis('off')

ax2.set_title('Regions')

for c_ax, c_prop in zip(n_axs[2:], grayco_prop_list):

    c_ax.imshow(prop_imgs[c_prop], cmap='gray')

    c_ax.axis('off')

    c_ax.set_title('{} Image'.format(c_prop))

import pandas as pd

out_df = pd.DataFrame(out_df_list)

out_df['positive_score'] = out_df['score'].map(

    lambda x: 'FG' if x > 0.35 else 'BG')

out_df.describe()
sns.pairplot(out_df,

             hue='positive_score',

             kind="reg")