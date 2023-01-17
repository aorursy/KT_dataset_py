import sys

import os

import random

import subprocess

from tqdm import tqdm



from six import string_types



# Make sure you have all of these packages installed, e.g. via pip

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import plotly.express as px

import scipy

from skimage import io

from scipy import ndimage

from IPython.display import display

%matplotlib inline
!ls -lha ../input/planets-dataset/planet/planet
PLANET_KAGGLE_ROOT = os.path.abspath("../input/planets-dataset/planet/planet")

PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')

PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_classes.csv')

assert os.path.exists(PLANET_KAGGLE_ROOT)

assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)

assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)
labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)

labels_df.head()
# Build list with unique labels

label_list = []

for tag_str in labels_df.tags.values:

    labels = tag_str.split(' ')

    for label in labels:

        if label not in label_list:

            label_list.append(label)
# Add onehot features for every label

for label in label_list:

    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

# Display head

labels_df.head()
# Histogram of label instances

labels_df[label_list].sum().sort_values().plot.bar()
def make_cooccurence_matrix(labels):

    numeric_df = labels_df[labels]; 

    c_matrix = numeric_df.T.dot(numeric_df)

    sns.heatmap(c_matrix, cmap ="Blues")

    return c_matrix

    

# Compute the co-ocurrence matrix

make_cooccurence_matrix(label_list)
weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']

make_cooccurence_matrix(weather_labels)
land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']

make_cooccurence_matrix(land_labels)
rare_labels = [l for l in label_list if labels_df[label_list].sum()[l] < 2000]

make_cooccurence_matrix(rare_labels)
for w in weather_labels:

    df_weather_subset = labels_df.loc[labels_df[w] == 1, label_list].drop([w], axis=1)

    weather_percent_subset = df_weather_subset.sum(axis =0) / df_weather_subset.shape[0]

    weather_percent_subset = weather_percent_subset[weather_percent_subset >0].sort_values(ascending=False)

    fig = px.bar(x=weather_percent_subset.index, y=weather_percent_subset.values,  

                 labels={'x':'label', 'y':f'Another labels given {w} label'})

    fig.update_layout(title_text=f"Main label: {w}", yaxis_tickformat=',.0%')

    fig.show()
def sample_images(tags, n=None):

    """Randomly sample n images with the specified tags."""

    condition = True

    if isinstance(tags, string_types):

        raise ValueError("Pass a list of tags, not a single tag.")

    for tag in tags:

        condition = condition & labels_df[tag] == 1

    if n is not None:

        return labels_df[condition].sample(n)

    else:

        return labels_df[condition]

    



def plot_rgbn_histo(r, g, b):

    for slice_, name, color in ((r,'r', 'red'),(g,'g', 'green'),(b,'b', 'blue')):

        plt.hist(slice_.ravel(), bins=100, 

                 range=[0,rgb_image.max()], 

                 label=name, color=color, histtype='step')

    plt.legend()
def load_image(filename):

    '''Look through the directory tree to find the image you specified

    (e.g. train_10.tif vs. train_10.jpg)'''

    for dirname in os.listdir(PLANET_KAGGLE_ROOT):

        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))

        if os.path.exists(path):

            #print('Found image {}'.format(path))

            return io.imread(path)

    # if you reach this line, you didn't find the image you're looking for

    print('Load failed: could not find image {}'.format(path))
def get_rgb_image(labels=['primary', 'water', 'road'], n_samples=1):

    s = sample_images(labels, n=n_samples)

    fnames = s.loc[:, "image_name"].apply(lambda fname: '{}.{}'.format(fname, "jpg"))

    rgb_images = []

    for name in fnames:

    # find the image in the data directory and load it

        bgr_image = load_image(name)

        rgb_image = bgr_image[:, :, [2,1,0]]

        rgb_images.append(rgb_image)

    return np.array(rgb_images)





def get_r_g_b_channels(rgb_image):

    b, g, r = rgb_image[:, :, 2], rgb_image[:, :, 1], rgb_image[:, :, 0]

    return r, g, b
rgb_images= get_rgb_image(labels=['primary', 'water', 'road'], n_samples=5)

rgb_image = rgb_images[0]

r, g, b = get_r_g_b_channels(rgb_image)

# plot a histogram of rgbn values

plot_rgbn_histo(r, g, b)
# Plot the bands

fig = plt.figure()

fig.set_size_inches(9, 3)

for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'))):

    a = fig.add_subplot(1, 3, i+1)

    a.set_title(c)

    plt.imshow(x)
plt.imshow(rgb_image)
all_image_paths = os.listdir(PLANET_KAGGLE_JPEG_DIR)

random.shuffle(all_image_paths)
n = 200



ref_colors = [[],[],[]]

for _file in tqdm(all_image_paths[:n]):

    # keep only the first 3 bands, RGB

    _img = mpimg.imread(os.path.join(PLANET_KAGGLE_JPEG_DIR, _file))[:,:,:3]

    # Flatten 2-D to 1-D

    _data = _img.reshape((-1,3))

    # Dump pixel values to aggregation buckets

    for i in range(3): 

        ref_colors[i] = ref_colors[i] + _data[:,i].tolist()

    

ref_colors = np.array(ref_colors)
ref_colors = np.array(ref_colors)

ref_color_mean = [np.mean(ref_colors[i]) for i in range(3)]

ref_color_std = [np.std(ref_colors[i]) for i in range(3)]
print("ref_color_mean:")

print(ref_color_mean)

print("ref_color_std:")

print(ref_color_std)
def calibrate_image(rgb_img):

    calibrated_img = rgb_image.copy().astype('float32')

    for i in range(3):

        calibrated_img[:,:,i] = (rgb_img[:,:,i] -  np.mean(rgb_img[:,:,i])) / np.std(rgb_img[:,:,i])

        calibrated_img[:,:,i] = calibrated_img[:,:,i] * ref_color_std[i] + ref_color_mean[i]

    return calibrated_img.astype('uint8')
img = calibrate_image(rgb_image)

plt.imshow(img)
def display_multiple_images(rgb_images):

    col, row = (1, len(rgb_images)) if len(rgb_images) <=4 else ((len(rgb_images) / 4) + 1, 4)

    fig = plt.figure()

    fig.set_size_inches(12, 3 * col)

    for i, _img in enumerate(rgb_images):

        a = fig.add_subplot(col, row, i+1)

        plt.imshow(calibrate_image(_img))
# provide labels to display sample images

labels = ['water']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)
# provide labels to display sample images

labels = ['primary']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)
# provide labels to display sample images

labels = ['agriculture']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)
# provide labels to display sample images

labels = ['cultivation']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)
# provide labels to display sample images

labels = ['habitation']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)
# provide labels to display sample images

labels = ['selective_logging']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)
# provide labels to display sample images

labels = ['slash_burn']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)
# provide labels to display sample images

labels = ['blow_down']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)
# provide labels to display sample images

labels = ['blooming']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)
# provide labels to display sample images

labels = ['conventional_mine']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)
# provide labels to display sample images

labels = ['artisinal_mine']

rgb_images= get_rgb_image(labels=labels, n_samples=4)

display_multiple_images(rgb_images)