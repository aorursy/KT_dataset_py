import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import shutil

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



print("Pandas Version:", pd.__version__)

print("Numpy Version:", np.__version__)

print("OpenCV Version:", cv2.__version__)
base_dir = "../input"



# Training dir/ids/imgs

train_dir = os.path.join(base_dir, "stage1_train")

train_ids = [path for path in os.listdir(train_dir)]

# Dictionary mappings from IDs to images and masks

train_imgs = dict([(ID, os.listdir(os.path.join(train_dir, ID, "images"))[0]) for ID in train_ids])

train_masks = dict([(ID, os.listdir(os.path.join(train_dir, ID, "masks"))) for ID in train_ids])

msk_cnt = 0

for msk in train_masks.values():

    msk_cnt += len(msk)



# Testing dir/ids/imgs

test_dir = os.path.join(base_dir, "stage1_test")

test_ids = [path for path in os.listdir(test_dir)]

# Dictionary mappings from IDs to images

test_imgs = dict([(ID, os.listdir(os.path.join(test_dir, ID, "images"))[0]) for ID in test_ids])

    

print("Number of train ID files:", len(train_ids))

print("Number of train images:", len(train_imgs))

print("Number of train masks:", msk_cnt)

print()

print("Number of test ID files:", len(test_ids))

print("Number of test images:", len(test_imgs))
def load_img_shapes(path_to_img):

    return cv2.imread(path_to_img).shape
def load_img(path_to_img):

    img = cv2.imread(path_to_img)

    return img
# Grab 1 example image and 8 example masks for that image

sample_img_id = train_ids[0]

sample_img_path = os.path.join(train_dir, sample_img_id, "images", train_imgs[sample_img_id])

sample_msk_paths = os.listdir(os.path.join(train_dir, sample_img_id, "masks"))[:8]



# Load image

img = load_img(sample_img_path)



# Plot image

plt.imshow(img)

plt.title(train_imgs[sample_img_id])



# Plot masks

plt.figure(figsize=(17, 10))

rows = 4

img_per_row = len(sample_msk_paths) // rows

for i in range(len(sample_msk_paths)):

    mask = load_img(os.path.join(train_dir, sample_img_id, "masks", sample_msk_paths[i]))

    plt.subplot(rows, img_per_row, i+1)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.imshow(mask)

    plt.title("Mask_" + str(i+1))

plt.show()



# Print number of image masks and mask names

print(len(sample_msk_paths), "Masks")

print("Mask image names: ", sample_msk_paths)
# Load distribution of training/testing image sizes

train_shapes = []

test_shapes = []

for i in range(len(train_imgs)):

    img_id = train_ids[i]

    img_path = os.path.join(train_dir, img_id, "images", train_imgs[img_id])

    train_shapes.append(load_img_shapes(img_path))

for i in range(len(test_imgs)):

    img_id = test_ids[i]

    img_path = os.path.join(test_dir, img_id, "images", test_imgs[img_id])

    test_shapes.append(load_img_shapes(img_path))



df_train = pd.DataFrame({'Shapes': train_shapes})

train_counts = df_train['Shapes'].value_counts()

df_test = pd.DataFrame({'Shapes': test_shapes})

test_counts = df_test['Shapes'].value_counts()

print("Training Image Shapes:")

for i in range(len(train_counts)):

    print("Shape %s counts: %d" % (train_counts.index[i], train_counts.values[i]))

print("*"*50)

print("Testing Image Shapes:")

for i in range(len(test_counts)):

    print("Shape %s counts: %d" % (test_counts.index[i], test_counts.values[i]))
# Plot distribution of train/test image shapes

plt.figure(figsize=(14, 10))

sns.barplot(x=train_counts.index, y=train_counts.values)

plt.title("Train Dataset")



plt.figure(figsize=(14, 10))

sns.barplot(x=test_counts.index, y=test_counts.values)

plt.title("Test Dataset")



plt.show()