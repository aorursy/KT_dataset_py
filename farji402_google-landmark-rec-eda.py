# Import required libs

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from skimage.transform import resize

from skimage.measure import find_contours

plt.style.use('ggplot')

import tensorflow as tf

import gc

from tqdm import tqdm



import warnings

warnings.filterwarnings('ignore')
# File paths 

train_img = np.array(tf.io.gfile.glob('../input/landmark-recognition-2020/train/*/*/*/*.jpg'))

test_img = np.array(tf.io.gfile.glob('../input/landmark-recognition-2020/test/*/*/*/*.jpg'))



print('There are %i images in train and %i images in test'%(len(train_img),len(test_img)))
# load labels

label = pd.read_csv('../input/landmark-recognition-2020/train.csv')
# Take a first look

print(label.info())

label.head()
# Show few images from both train and test

def plot_img(img_files, show_label= False):

    """Show 9 images from img_files"""

    plt.figure(figsize=(10, 10))

    for i, img_file in enumerate(np.random.choice(img_files, size= np.min([len(img_files), 9]), replace= False)):

        ax = plt.subplot(3, 3, i + 1)

        img = plt.imread(img_file)

        img = resize(img, (256, 256), anti_aliasing= True)

        plt.imshow(img)

        plt.axis("off")

        if show_label:

            img_name = img_file.split('/')[-1].split('.')[-2]

            img_label = label[label['id'] == img_name]['landmark_id'].values[0]

            plt.title('Class: ' + str(img_label))

        # save memory

        del(img)

        gc.collect()
# Train images

print('TRAIN IMAGES')

plot_img(train_img, True)
# Test images

print('TEST IMAGES')

plot_img(test_img)
# Checking duplicates

print('Fraction of unique train images: ', len(label['id'].unique())/len(label))

print('Total number of classes: ', len(label['landmark_id'].unique()))
# Class distribution

label['landmark_id'].astype('category').describe()
# top class distribution

fig, ax = plt.subplots(figsize= (12,6))

top_class = label['landmark_id'].value_counts().iloc[:30].reset_index()

sns.barplot(x= 'index', y= 'landmark_id', data= top_class, ax= ax, palette= 'rocket')

fig.autofmt_xdate()

plt.xlabel('Image Class')

plt.ylabel('Frequency')

plt.margins(0.05)
# Find contours in images

"""cont_train = pd.DataFrame({

    'contours': np.zeros(50000)})

labels= []

random_sample = np.random.choice(train_img, size= 50000, replace= False)

for i, img_file in tqdm(enumerate(random_sample)):

    img = plt.imread(img_file)

    img = tf.image.rgb_to_grayscale(img)

    img = img.numpy().reshape((img.shape[0], img.shape[1]))

    cont_train['contours'][i] = len(find_contours(img, level= 100))

    labels.append(label[label['id'] == img_file.split('/')[-1].split('.')[0]]['landmark_id'])

    # save memory

    del(img)

    gc.collect()



cont_train['label'] = labels"""
cont_train = pd.read_csv('../input/train-data-contours/train_contours.csv')

cont_train.head()
counts = label['landmark_id'].value_counts().sort_values(ascending= False)



top_class = counts.index[:50].values

top_class_df = cont_train[cont_train['label'].isin(top_class).values]



fig, ax = plt.subplots(figsize= (16,4))

sns.pointplot(x= 'label', y= 'contours', data= top_class_df, ax= ax)

fig.autofmt_xdate()

plt.show()
img_name = train_img[13].split('/')[-1].split('.')[0]

img = label[label['id'] == img_name]

same_imgs = label[label['landmark_id'] == img['landmark_id'].values[0]]

for i in same_imgs['id']:

    img_path = '../input/landmark-recognition-2020/train/' + i[0] + '/' + i[1] + '/' + i[2] + '/' + i + '.jpg'

    img = plt.imread(img_path)

    img = tf.image.rgb_to_grayscale(img)

    img = img.numpy().reshape((img.shape[0], img.shape[1]))

    conts = find_contours(img, level= 100)



    fig, ax = plt.subplots(1,2, figsize= (10,10))

    ax[0].imshow(img)

    ax[0].axis('off')



    ax[1].imshow(img)

    for cont in conts:

        ax[1].plot([val[1] for val in cont], [val[0] for val in cont])

        ax[1].set_title('Number of Contours %i'%len(conts))

    plt.axis('off')

    plt.show()
# Show contours in a test image

img = plt.imread(test_img[4])

img = tf.image.rgb_to_grayscale(img)

img = img.numpy().reshape((img.shape[0], img.shape[1]))

conts = find_contours(img, level= 100)



fig, ax = plt.subplots(1,2, figsize= (20,20))

ax[0].imshow(img)

ax[0].axis('off')



ax[1].imshow(img)

for cont in conts:

    ax[1].plot([val[1] for val in cont], [val[0] for val in cont])

plt.axis('off')

plt.show()
top_class = label['landmark_id'].value_counts().reset_index()

top_class.head()
freq_img = label[label['landmark_id'] == 126637][:4].reset_index()



def make_path(image_name):

    img_path = '../input/landmark-recognition-2020/train/' + image_name[0] + '/' + image_name[1] + '/' + image_name[2] + '/' + image_name + '.jpg'

    return img_path

freq_img['path'] = freq_img['id'].apply(make_path)

plot_img(freq_img['path'].values, True)
freq_img = label[label['landmark_id'] == 126637][:4].reset_index()



def make_path(image_name):

    img_path = '../input/landmark-recognition-2020/train/' + image_name[0] + '/' + image_name[1] + '/' + image_name[2] + '/' + image_name + '.jpg'

    return img_path

freq_img['path'] = freq_img['id'].apply(make_path)

plot_img(freq_img['path'].values, True)
freq_img = label[label['landmark_id'] == 20409][:4].reset_index()



def make_path(image_name):

    img_path = '../input/landmark-recognition-2020/train/' + image_name[0] + '/' + image_name[1] + '/' + image_name[2] + '/' + image_name + '.jpg'

    return img_path

freq_img['path'] = freq_img['id'].apply(make_path)

plot_img(freq_img['path'].values, True)