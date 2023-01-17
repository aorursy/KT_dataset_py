DATASET_DIR = '../input/severstal-steel-defect-detection/'

TEST_SIZE = 0.3

RANDOM_STATE = 123



NUM_TRAIN_SAMPLES = 20 # The number of train samples used for visualization

NUM_VAL_SAMPLES = 20 # The number of val samples used for visualization

COLORS = ['b', 'g', 'r', 'm'] # Color of each class
import pandas as pd

import os

import cv2

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection



from shutil import copyfile

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook
df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
df.head()
legacy_df = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])



for img_id, img_df in tqdm_notebook(df.groupby('ImageId')):

    for i in range(1, 5):

        avail_classes = list(img_df.ClassId)



        row = dict()

        row['ImageId_ClassId'] = img_id + '_' + str(i)



        if i in avail_classes:

            row['EncodedPixels'] = img_df.loc[img_df.ClassId == i].EncodedPixels.iloc[0]

        else:

            row['EncodedPixels'] = np.nan

        

        legacy_df = legacy_df.append(row, ignore_index=True)
legacy_df.head()
df = legacy_df
df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])

df['HavingDefection'] = df['EncodedPixels'].map(lambda x: 0 if x is np.nan else 1)
image_col = np.array(df['Image'])

image_files = image_col[::4]

all_labels = np.array(df['HavingDefection']).reshape(-1, 4)



num_img_class_1 = np.sum(all_labels[:, 0])

num_img_class_2 = np.sum(all_labels[:, 1])

num_img_class_3 = np.sum(all_labels[:, 2])

num_img_class_4 = np.sum(all_labels[:, 3])

print('Class 1: {} images'.format(num_img_class_1))

print('Class 2: {} images'.format(num_img_class_2))

print('Class 3: {} images'.format(num_img_class_3))

print('Class 4: {} images'.format(num_img_class_4))
def plot_figures(

    sizes,

    pie_title,

    start_angle,

    bar_title,

    bar_ylabel,

    labels=('Class 1', 'Class 2', 'Class 3', 'Class 4'),

    colors=None,

    explode=(0, 0, 0, 0.1),

):

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))



    y_pos = np.arange(len(labels))

    barlist = axes[0].bar(y_pos, sizes, align='center')

    axes[0].set_xticks(y_pos, labels)

    axes[0].set_ylabel(bar_ylabel)

    axes[0].set_title(bar_title)

    if colors is not None:

        for idx, item in enumerate(barlist):

            item.set_color(colors[idx])



    def autolabel(rects):

        """

        Attach a text label above each bar displaying its height

        """

        for rect in rects:

            height = rect.get_height()

            axes[0].text(

                rect.get_x() + rect.get_width()/2., height,

                '%d' % int(height),

                ha='center', va='bottom', fontweight='bold'

            )



    autolabel(barlist)

    

    pielist = axes[1].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=start_angle, counterclock=False)

    axes[1].axis('equal')

    axes[1].set_title(pie_title)

    if colors is not None:

        for idx, item in enumerate(pielist[0]):

            item.set_color(colors[idx])



    plt.show()
print('[THE WHOLE DATASET]')



sum_each_class = np.sum(all_labels, axis=0)

plot_figures(

    sum_each_class,

    pie_title='The percentage of each class',

    start_angle=90,

    bar_title='The number of images for each class',

    bar_ylabel='Images',

    colors=COLORS,

    explode=(0, 0, 0, 0.1)

)



sum_each_sample = np.sum(all_labels, axis=1)

unique, counts = np.unique(sum_each_sample, return_counts=True)



plot_figures(

    counts,

    pie_title='The percentage of the number of classes appears in an image',

    start_angle=120,

    bar_title='The number of classes appears in an image',

    bar_ylabel='Images',

    labels=list(unique),

    explode=np.zeros(len(unique))

)
annotations = np.array(df['EncodedPixels']).reshape(-1, 4)
X_train, X_val, y_train, y_val = train_test_split(image_files, annotations, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print('X_train:', X_train.shape)

print('y_train:', y_train.shape)

print('X_val:', X_val.shape)

print('y_val', y_val.shape)
print('[TRAINING SET]')



tmp = y_train.reshape(-1)

tmp = list(map(lambda x: 0 if x is np.nan else 1, tmp))

train_labels = np.array(tmp).reshape(-1, 4)



sum_each_class = np.sum(train_labels, axis=0)

plot_figures(

    sum_each_class,

    pie_title='The percentage of each class',

    start_angle=90,

    bar_title='The number of images for each class',

    bar_ylabel='Images',

    colors=COLORS,

    explode=(0, 0, 0, 0.1)

)





sum_each_sample = np.sum(train_labels, axis=1)

unique, counts = np.unique(sum_each_sample, return_counts=True)



plot_figures(

    counts,

    pie_title='The percentage of the number of classes appears in an image',

    start_angle=120,

    bar_title='The number of classes appears in an image',

    bar_ylabel='Images',

    labels=list(unique),

    explode=np.zeros(len(unique))

)
print('[VALIDATION SET]')



tmp = y_val.reshape(-1)

tmp = list(map(lambda x: 0 if x is np.nan else 1, tmp))

val_labels = np.array(tmp).reshape(-1, 4)



sum_each_class = np.sum(val_labels, axis=0)

plot_figures(

    sum_each_class,

    pie_title='The percentage of each class',

    start_angle=90,

    bar_title='The number of images for each class',

    bar_ylabel='Images',

    colors=COLORS,

    explode=(0, 0, 0, 0.1)

)





sum_each_sample = np.sum(val_labels, axis=1)

unique, counts = np.unique(sum_each_sample, return_counts=True)



plot_figures(

    counts,

    pie_title='The percentage of the number of classes appears in an image',

    start_angle=120,

    bar_title='The number of classes appears in an image',

    bar_ylabel='Images',

    labels=list(unique),

    explode=np.zeros(len(unique))

)
def rle2mask(mask_rle, shape=(1600,256)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T
def show_samples(samples):

    for sample in samples:

        fig, ax = plt.subplots(figsize=(16, 10))

        img_path = os.path.join(DATASET_DIR, 'train_images', sample[0])

        img = cv2.imread(img_path)



        patches = []

        for idx, rle in enumerate(sample[1]):

            if rle is not np.nan:

                mask = rle2mask(rle)

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:

                    poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=1, edgecolor=COLORS[idx], facecolor=COLORS[idx], fill=True)

                    patches.append(poly_patch)

        p = PatchCollection(patches, match_original=True, cmap=matplotlib.cm.jet, alpha=0.3)



        ax.imshow(img/255)

        ax.set_title(sample[0])

        ax.add_collection(p)

        ax.set_xticklabels([])

        ax.set_yticklabels([])

        plt.show()
train_pairs = np.array(list(zip(X_train, y_train)))

train_samples = train_pairs[np.random.choice(train_pairs.shape[0], NUM_TRAIN_SAMPLES, replace=False), :]



show_samples(train_samples)
val_pairs = np.array(list(zip(X_val, y_val)))

val_samples = val_pairs[np.random.choice(val_pairs.shape[0], NUM_VAL_SAMPLES, replace=False), :]



show_samples(val_samples)
!mkdir train_images

!mkdir val_images
for image_file in tqdm_notebook(X_train):

    src = os.path.join(DATASET_DIR, 'train_images', image_file)

    dst = os.path.join('./train_images', image_file)

    copyfile(src, dst)



for image_file in tqdm_notebook(X_val):

    src = os.path.join(DATASET_DIR, 'train_images', image_file)

    dst = os.path.join('./val_images', image_file)

    copyfile(src, dst)
!apt install zip
!zip -r -m -1 -q train_images.zip ./train_images

!zip -r -m -1 -q val_images.zip ./val_images
y_train = y_train.reshape(-1)

y_val = y_val.reshape(-1)
X_train = np.repeat(X_train, 4)

X_val = np.repeat(X_val, 4)



X_train = X_train.reshape(-1, 4)

X_val = X_val.reshape(-1, 4)



indices = np.array(['_1', '_2', '_3', '_4'])



X_train += indices

X_val += indices



X_train = X_train.reshape(-1)

X_val = X_val.reshape(-1)
train_set = {

    'ImageId_ClassId': X_train,

    'EncodedPixels': y_train

}



val_set = {

    'ImageId_ClassId': X_val,

    'EncodedPixels': y_val

}



train_df = pd.DataFrame(train_set)

val_df = pd.DataFrame(val_set)



train_df.to_csv('./train.csv', index=False)

val_df.to_csv('./val.csv', index=False)