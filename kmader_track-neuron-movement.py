import os

from skimage.io import imread

from glob import glob

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
with np.load('../input/flo_image_1.npz') as im_data:

    image_dict = dict(zip(im_data['image_ids'], im_data['image_stack']))

print('Loaded',len(image_dict), 'images')
# show a few test images

fig, m_axs = plt.subplots(2,3)

for (ax1, ax2), (img_name, img_data) in  zip(m_axs.T, image_dict.items()):

    ax1.imshow(img_data)

    ax1.set_title('{}\n{}'.format(img_name, img_data.shape))

    ax2.hist(img_data.ravel())
time_df = pd.read_csv('../input/data141110.csv')

time_df['path'] = time_df['Image.No.'].map(lambda x: "141110A3.%04d" % (x))

time_df['loaded'] = time_df['path'].map(lambda x: x in image_dict)

valid_time_df = time_df.query('loaded')

valid_time_df.sample(3)
bg_image=np.max(np.stack(image_dict.values(),0),0) # create an image for normalizing the values
from skimage.filters import threshold_otsu as thresh_func

# show a few test images

fig, m_axs = plt.subplots(3,2, figsize = (12, 8))

for (ax1, ax2, ax3), (img_name, rimg_data) in  zip(m_axs.T, image_dict.items()):

    img_data =np.log10(rimg_data/bg_image)

    ax1.imshow(img_data)

    ax1.set_title(img_name)

    thresh_val = thresh_func(img_data)

    ax2.hist(img_data.ravel(),50)

    ax2.axvline(thresh_val, color='r')

    ax3.imshow(img_data>thresh_val)
from skimage.feature import blob_dog, blob_log, blob_doh # three different blob functions

# show a few test images

blog_kwargs = dict(min_sigma = 2.5, max_sigma=5)

fig, m_axs = plt.subplots(3,3, figsize = (12, 8))

for n_axs, (img_name, rimg_data) in zip(m_axs, image_dict.items()):

    img_data = np.log10(rimg_data/bg_image)[-128:,:128]

    # compute the blobs

    image_gray = (img_data-img_data.mean())/img_data.std() # normalize data

    image_gray = image_gray.clip(-1,1)

    

    blobs_log = blob_log(image_gray,num_sigma=3, threshold=0.2, **blog_kwargs)



    # Compute radii in the 3rd column.

    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)



    blobs_dog = blob_dog(image_gray, threshold=.2,**blog_kwargs)

    blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)



    blobs_doh = blob_doh(image_gray, threshold=.02,**blog_kwargs)

    blobs_list = [blobs_dog, blobs_log, blobs_doh]

    

    for c_ax, c_blobs, blob_func in zip(n_axs, blobs_list, [blob_dog, blob_log, blob_doh]):

        c_ax.imshow(img_data, cmap = 'bone')

        c_ax.set_title('{}\n{}'.format(img_name,blob_func.__name__))

        for blob in c_blobs:

            y, x, r = blob

            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)

            c_ax.add_patch(c)

        c_ax.set_axis_off()

    
def generate_blob_df():

    for idx, (img_name, rimg_data) in enumerate(image_dict.items()):

        # preprocess the image

        img_data = np.log10(rimg_data/bg_image)

        

        image_gray = (img_data-img_data.mean())/img_data.std() # normalize data

        image_gray = image_gray.clip(-1,1)

        # compute blobs

        blobs_dog = blob_dog(image_gray, threshold=.2,**blog_kwargs)

        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

        # format output as dataframe

        cblob_df = pd.DataFrame(blobs_dog, columns=['x', 'y', 'r'])

        cblob_df['frame'] = idx

        cblob_df['img_name'] = img_name

        cblob_df['blob_idx'] = range(cblob_df.shape[0])

        yield cblob_df
%%time

blob_series_df = pd.concat([blobs for _, blobs in zip(range(5), # only keep 10 frames

                              generate_blob_df())]

                          ).query('r>4.0') # only keep blobs larger than 2 pixels
print('Found',blob_series_df.shape[0],'blobs')

blob_series_df.sample(4)
# match each frame to the next frame

blob_series_p1_df = blob_series_df.copy()

blob_series_p1_df['frame'] = blob_series_df['frame'].map(lambda x: x-1)

blob_n_np1_df = pd.merge(blob_series_df, blob_series_p1_df, on=['frame'], suffixes=('','_next'))

print(blob_n_np1_df.shape[0], 'blob pairs')
# calculate the distance from a current blob to a blob in the next frame

blob_n_np1_df['distance'] = blob_n_np1_df.apply(lambda row: np.sqrt(np.power(row['x']-row['x_next'],2)+

                                                                    np.power(row['y']-row['y_next'],2)),1)

blob_n_np1_df.sample(3)
# find the nearest distance blobs

min_dist_idx = blob_n_np1_df.groupby(['frame', 'blob_idx']).apply(lambda df_grp: df_grp['distance'].argmin())

blob_track_df = blob_n_np1_df.ix[min_dist_idx, :]

blob_track_df.sample(4)
# show the matches on the first frame

ff_df = blob_track_df.query('frame==0')

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 3))

ax1.plot(ff_df.x, ff_df.y, 'ro', label='N=0')

ax1.plot(ff_df.x_next, ff_df.y_next, 'g.', label='N=1')

ax1.legend()

ax1.set_title('Blob Positions')



ax2.quiver(ff_df.x, ff_df.y, ff_df.x_next-ff_df.x, ff_df.y_next-ff_df.y, pivot='mid', units='x')

ax2.set_title('Offsets')



ax3.quiver(ff_df.x*0, ff_df.y*0, ff_df.x_next-ff_df.x, ff_df.y_next-ff_df.y, pivot='mid', units='x')

ax3.set_title('Blob Movements')