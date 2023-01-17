%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.rcParams["figure.figsize"] = (6, 6)

plt.rcParams["figure.dpi"] = 200

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'viridis'
from pathlib import Path

import pandas as pd

import numpy as np

from PIL import Image

from skimage.io import imread

from skimage.util import montage

from tqdm import tqdm

tqdm.pandas() # hack progressbars into pandas

montage_rgb = lambda x, **kwargs: np.stack([montage(x[:, :, :, i], **kwargs) for i in range(x.shape[3])], -1)
satellite_dir = Path('../input/satellite-images-of-hurricane-damage/')

image_df = pd.DataFrame({'path': list(satellite_dir.glob('**/*.jp*g'))})

image_df.sample(3)
image_df['damage'] = image_df['path'].map(lambda x: x.parent.stem)

image_df['data_split'] = image_df['path'].map(lambda x: x.parent.parent.stem)

image_df['location'] = image_df['path'].map(lambda x: x.stem)

image_df['lat'] = image_df['location'].map(lambda x: float(x.split('_')[0]))

image_df['lon'] = image_df['location'].map(lambda x: float(x.split('_')[-1]))

image_df.sample(3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for c_group, c_rows in image_df.groupby(['damage']):

    ax1.plot(c_rows['lat'], c_rows['lon'], '.', label=c_group, alpha=0.5)

ax1.legend()

ax1.set_title('Data by Damage Type')

for c_group, c_rows in image_df.groupby(['data_split']):

    ax2.plot(c_rows['lat'], c_rows['lon'], '.', label=c_group, alpha=0.5)

ax2.legend()

ax2.set_title('Data by Group')
fig, m_axs = plt.subplots(1, 2, figsize=(20, 10))

for c_ax, (c_cat, c_rows) in zip(m_axs, image_df.groupby(['damage'])):

    img_stack = np.stack(c_rows.sample(121)['path'].map(imread), 0)

    c_ax.imshow(montage_rgb(img_stack))

    c_ax.set_title(c_cat)

    c_ax.axis('off')
fig, m_axs = plt.subplots(2, 2, figsize=(20, 20))

for c_ax, (c_cat, c_rows) in zip(m_axs.flatten(), image_df.groupby(['data_split'])):

    img_stack = np.stack(c_rows.sample(121)['path'].map(imread), 0)

    c_ax.imshow(montage_rgb(img_stack))

    c_ax.set_title(c_cat)

    c_ax.axis('off')
test_image = Image.open(image_df['path'].iloc[1010]) # normal image

# convert to 8bit color (animated GIF) and then back

web_image = test_image.convert('P', palette='WEB', dither=None)

few_color_image = web_image.convert('RGB')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(test_image)

ax2.imshow(few_color_image)
print('Unique colors before', len(set([tuple(rgb) for rgb in np.array(test_image).reshape((-1, 3))])))

print('Unique colors after', len(set([tuple(rgb) for rgb in np.array(few_color_image).reshape((-1, 3))])))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

for c_channel, c_name in enumerate(['red', 'green', 'blue']):

    ax1.hist(np.array(test_image)[:, :, c_channel].ravel(), 

             color=c_name[0], 

             label=c_name, 

             bins=np.arange(256), 

             alpha=0.5)

    ax2.hist(np.array(few_color_image)[:, :, c_channel].ravel(), 

             color=c_name[0], 

             label=c_name, 

             bins=np.arange(256), 

             alpha=0.5)
idx_to_color = np.array(web_image.getpalette()).reshape((-1, 3))/255.0
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

ax1.imshow(few_color_image)

counts, bins = np.histogram(web_image, bins=np.arange(256))

for i in range(counts.shape[0]):

    ax2.bar(bins[i], counts[i], color=idx_to_color[i])

ax2.set_yscale('log')

ax2.set_xlabel('Color Id')

ax2.set_ylabel('Pixel Count')
def color_count_feature(in_path):

    raw_image = Image.open(in_path) 

    web_image = raw_image.convert('P', palette='WEB', dither=None)

    counts, bins = np.histogram(np.array(web_image).ravel(), bins=np.arange(256))

    return counts*1.0/np.prod(web_image.size) # normalize output
%%time

image_df['color_features'] = image_df['path'].progress_map(color_count_feature)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

combined_features = np.stack(image_df['color_features'].values, 0)

ax1.imshow(combined_features)

ax1.set_title('Raw Color Counts')

ax1.set_xlabel('Color')

ax1.set_ylabel('Frequency')

ax1.set_aspect(0.01)

color_wise_average = np.tile(np.mean(combined_features, 0, keepdims=True), (combined_features.shape[0], 1)).clip(1/(128*128), 1)

ax2.imshow(combined_features/color_wise_average, vmin=0.05, vmax=20)

ax2.set_title('Normalized Color Counts')

ax2.set_xlabel('Color')

ax2.set_ylabel('Frequency')

ax2.set_aspect(0.01)
from sklearn.decomposition import PCA

xy_pca = PCA(n_components=2)

xy_coords = xy_pca.fit_transform(combined_features)

image_df['x'] = xy_coords[:, 0]

image_df['y'] = xy_coords[:, 1]
fig, ax1 = plt.subplots(1,1, figsize=(15, 15))

for c_group, c_row in image_df.groupby('damage'):

    ax1.plot(c_row['x'], c_row['y'], '*', label=c_group)

ax1.legend()

def show_xy_images(in_df, image_zoom=1):

    fig, ax1 = plt.subplots(1,1, figsize=(10, 10))

    artists = []

    for _, c_row in in_df.iterrows():

        c_img = Image.open(c_row['path']).resize((64, 64))

        img = OffsetImage(c_img, zoom=image_zoom)

        ab = AnnotationBbox(img, (c_row['x'], c_row['y']), xycoords='data', frameon=False)

        artists.append(ax1.add_artist(ab))

    ax1.update_datalim(in_df[['x', 'y']])

    ax1.autoscale()

    ax1.axis('off')

show_xy_images(image_df.sample(200))
image_df['path'] = image_df['path'].map(str) # saving pathlib objects causes problems

image_df.to_json('color_features.json')
image_df.sample(3)
from keras.applications import resnet50

from keras import models, layers

pretrained_model = resnet50.ResNet50(include_top=False, weights='imagenet')

feature_model = models.Sequential(name='just_features')

prep_layer = layers.Conv2D(3, 

                           kernel_size=(1, 1), 

                           weights=[np.expand_dims(np.expand_dims(np.eye(3), 0), 0), 

                                    np.array([-103.9, -116.78, -123.68])],

                           input_shape=(None, None, 3),

                          name='PreprocessingLayer')

feature_model.add(prep_layer)

feature_model.add(pretrained_model)

feature_model.add(layers.GlobalAveragePooling2D())

feature_model.save('feature_model.h5')

feature_model.summary()
%%time

image_df['resnet_features'] = image_df['path'].progress_map(lambda x: feature_model.predict(np.expand_dims(imread(x), 0))[0])
image_df.to_json('resnet_features.json')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

combined_features = np.stack(image_df['resnet_features'].values, 0)

ax1.imshow(combined_features)

ax1.set_title('Raw Feature Values')

ax1.set_xlabel('Color')

ax1.set_ylabel('Frequency')

ax1.set_aspect(0.01)

color_wise_average = np.tile(np.mean(combined_features, 0, keepdims=True), (combined_features.shape[0], 1))

color_wise_std = np.tile(np.std(combined_features, 0, keepdims=True), (combined_features.shape[0], 1)).clip(0.01, 10)

ax2.imshow((combined_features-color_wise_average)/color_wise_std, vmin=-2, vmax=2, cmap='RdBu')

ax2.set_title('Normalized Feature Values')

ax2.set_xlabel('Color')

ax2.set_ylabel('Frequency')

ax2.set_aspect(0.01)
from sklearn.decomposition import PCA

xy_pca = PCA(n_components=2)

xy_coords = xy_pca.fit_transform(combined_features)

image_df['x'] = xy_coords[:, 0]

image_df['y'] = xy_coords[:, 1]
fig, ax1 = plt.subplots(1,1, figsize=(15, 15))

for c_group, c_row in image_df.groupby('damage'):

    ax1.plot(c_row['x'], c_row['y'], '*', label=c_group)

ax1.legend()
show_xy_images(image_df.groupby('damage').apply(lambda x: x.sample(100)))