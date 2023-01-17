import sys

import os

import subprocess



from six import string_types



# Make sure you have all of these packages installed, e.g. via pip

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import scipy

from skimage import io

from scipy import ndimage

from IPython.display import display

%matplotlib inline
!ls -lha ../input
!ls -lha ../input/test-tif-v2 | wc -l
PLANET_KAGGLE_ROOT = os.path.abspath("../input/")

PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')

PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')

assert os.path.exists(PLANET_KAGGLE_ROOT)

assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)

assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)
!ls -lha /kaggle/input/train_v2.csv
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

    sns.heatmap(c_matrix)

    return c_matrix

    

# Compute the co-ocurrence matrix

make_cooccurence_matrix(label_list)
weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']

make_cooccurence_matrix(weather_labels)
land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']

make_cooccurence_matrix(land_labels)
rare_labels = [l for l in label_list if labels_df[label_list].sum()[l] < 2000]

make_cooccurence_matrix(rare_labels)
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
def load_image(filename):

    '''Look through the directory tree to find the image you specified

    (e.g. train_10.tif vs. train_10.jpg)'''

    for dirname in os.listdir(PLANET_KAGGLE_ROOT):

        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))

        if os.path.exists(path):

            print('Found image {}'.format(path))

            return io.imread(path)

    # if you reach this line, you didn't find the image you're looking for

    print('Load failed: could not find image {}'.format(path))

    

def sample_to_fname(sample_df, row_idx, suffix='tif'):

    '''Given a dataframe of sampled images, get the

    corresponding filename.'''

    fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')

    return '{}.{}'.format(fname, suffix)
def plot_rgbn_histo(r, g, b, n):

    for slice_, name, color in ((r,'r', 'red'),(g,'g', 'green'),(b,'b', 'blue'), (nir, 'nir', 'magenta')):

        plt.hist(slice_.ravel(), bins=100, 

                 range=[0,rgb_image.max()], 

                 label=name, color=color, histtype='step')

    plt.legend()
s = sample_images(['primary', 'water', 'road'], n=1)

fname = sample_to_fname(s, 0)



# find the image in the data directory and load it

# note the initial bgrn band ordering

bgrn_image = load_image(fname)



# extract the rgb values

bgr_image = bgrn_image[:,:,:3]

rgb_image = bgr_image[:, :, [2,1,0]]



# extract the different bands

b, g, r, nir = bgrn_image[:, :, 0], bgrn_image[:, :, 1], bgrn_image[:, :, 2], bgrn_image[:, :, 3]



# plot a histogram of rgbn values

plot_rgbn_histo(r, g, b, nir)
# Plot the bands

fig = plt.figure()

fig.set_size_inches(12, 4)

for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'), (nir, 'near-ir'))):

    a = fig.add_subplot(1, 4, i+1)

    a.set_title(c)

    plt.imshow(x)
plt.imshow(rgb_image)
# Pull a list of 20000 image names

jpg_list = os.listdir(PLANET_KAGGLE_JPEG_DIR)[:20000]

# Select a random sample of 100 among those

np.random.shuffle(jpg_list)

jpg_list = jpg_list[:100]
print(jpg_list)
ref_colors = [[],[],[]]

for _file in jpg_list:

    # keep only the first 3 bands, RGB

    _img = mpimg.imread(os.path.join(PLANET_KAGGLE_JPEG_DIR, _file))[:,:,:3]

    # Flatten 2-D to 1-D

    _data = _img.reshape((-1,3))

    # Dump pixel values to aggregation buckets

    for i in range(3): 

        ref_colors[i] = ref_colors[i] + _data[:,i].tolist()

    

ref_colors = np.array(ref_colors)
for i,color in enumerate(['r','g','b']):

    plt.hist(ref_colors[i], bins=30, range=[0,255], label=color, color=color, histtype='step')

plt.legend()

plt.title('Reference color histograms')
ref_means = [np.mean(ref_colors[i]) for i in range(3)]

ref_stds = [np.std(ref_colors[i]) for i in range(3)]
def calibrate_image(rgb_image):

    # Transform test image to 32-bit floats to avoid 

    # surprises when doing arithmetic with it 

    calibrated_img = rgb_image.copy().astype('float32')



    # Loop over RGB

    for i in range(3):

        # Subtract mean 

        calibrated_img[:,:,i] = calibrated_img[:,:,i]-np.mean(calibrated_img[:,:,i])

        # Normalize variance

        calibrated_img[:,:,i] = calibrated_img[:,:,i]/np.std(calibrated_img[:,:,i])

        # Scale to reference 

        calibrated_img[:,:,i] = calibrated_img[:,:,i]*ref_stds[i] + ref_means[i]

        # Clip any values going out of the valid range

        calibrated_img[:,:,i] = np.clip(calibrated_img[:,:,i],0,255)



    # Convert to 8-bit unsigned int

    return calibrated_img.astype('uint8')
test_image_calibrated = calibrate_image(rgb_image)

for i,color in enumerate(['r','g','b']):

    plt.hist(test_image_calibrated[:,:,i].ravel(), bins=30, range=[0,255], 

             label=color, color=color, histtype='step')

plt.legend()

plt.title('Calibrated image color histograms')
plt.imshow(test_image_calibrated)
sampled_images = sample_images(['clear', 'road', 'water'], n=3)



for i in range(len(sampled_images)):

    tif = sample_to_fname(sampled_images, i, 'tif')

    jpg = sample_to_fname(sampled_images, i, 'jpg')



    try:

        tif_img = load_image(tif)[:,:,:3]

        jpg_img = load_image(jpg)[:,:,:3]



        fig = plt.figure()

        plt.imshow(calibrate_image(tif_img))



        fig = plt.figure()

        plt.imshow(calibrate_image(jpg_img))

    except:

        continue

        

        
fig = plt.figure()

fig.set_size_inches(12, 4)

for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'), (nir, 'near-ir'))):

    a = fig.add_subplot(1, 4, i+1)

    a.set_title(c)

    plt.imshow(x)
rotated = scipy.ndimage.rotate(rgb_image, angle=45)

plt.imshow(rgb_image)

plt.imshow(calibrate_image(rotated))

rotated.shape