import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import cv2

import re

import albumentations as A

%matplotlib inline
INPUT_DIR = '../input/global-wheat-detection'

TRAIN_DIR = f'{INPUT_DIR}/train'
sample = cv2.imread(TRAIN_DIR+'/01189a3c3.jpg', cv2.IMREAD_UNCHANGED)



dimensions = sample.shape

height = sample.shape[0]

width = sample.shape[1]

n_of_channels = sample.shape[2]



print('Image characteristics:\n')

print('Dimensions: {}\nHeight: {}, Width:{}, Number of channels: {}'.format(dimensions, height, width, n_of_channels))
train_df = pd.read_csv(INPUT_DIR+'/train.csv')

train_df.head()
print(train_df['height'].unique(), train_df['width'].unique())
# No of unique images in the csv file



print('Unique images: ',len(train_df['image_id'].unique()))



# Different regions for which data is collected



print('Regions: ',train_df['source'].unique())



# No of unique images for each region

region_list = []

unique_images = []

for region in train_df['source'].unique():

    region_list.append(region)

    unique_images.append(len(train_df[train_df['source']== region]['image_id'].unique()))

    print('Region: {}, Number of Images: {}'.format(str(region), len(train_df[train_df['source']== region]['image_id'].unique())))
fig, ax = plt.subplots()

ax.pie(unique_images, labels = region_list, autopct='%1.1f%%')

ax.axis('equal')

plt.show()
train_df['x_min'] = -1

train_df['y_min'] = -1

train_df['w'] = -1

train_df['h'] = -1



def expand_bbox(x):

    r = np.array(re.findall('([0-9]+[.]?[0-9]*)', x))

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r



train_df[['x_min', 'y_min', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

train_df.drop(columns = ['bbox'], inplace = True)

train_df['x_min'] = train_df['x_min'].astype(np.float)

train_df['y_min'] = train_df['y_min'].astype(np.float)

train_df['w'] = train_df['w'].astype(np.float)

train_df['h'] = train_df['h'].astype(np.float)



train_df['x_max'] = train_df['x_min'] + train_df['w']

train_df['y_max'] = train_df['y_min'] + train_df['h']



train_df.head()
train_df.drop(columns = ['width', 'height'], inplace=True)

train_df.head()
def show_sample_images(image_data):

    

    fig, ax = plt.subplots(1, 2, figsize = (12, 8))

    ax = ax.flatten()

    

    image = cv2.imread(os.path.join(TRAIN_DIR + '/{}.jpg').format(image_data.iloc[0]['image_id']), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)

    

    ax[0].set_title('Original Image')

    ax[0].imshow(image)

    

    for i, row in image_data.iterrows():

        cv2.rectangle(image,

                      (int(row['x_min']), int(row['y_min'])),

                      (int(row['x_max']), int(row['y_max'])),

                      (220, 0, 0), 3)

    

    ax[1].set_title('Image with Bounding Boxes')

    ax[1].imshow(image)

    

    plt.show()

        
show_sample_images(train_df[train_df['image_id'] == 'b6ab77fd7'])
def get_bboxes(bboxes, col, bbox_format = 'pascal_voc', color='white'):

    for i in range(len(bboxes)):

        x_min = bboxes[i][0]

        y_min = bboxes[i][1]

        x_max = bboxes[i][2]

        y_max = bboxes[i][3]

        width = x_max - x_min

        height = y_max - y_min

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')

        col.add_patch(rect)
def show_augmented_images(aug_result, image_data):

    

    fig, ax = plt.subplots(1, 2, figsize = (12, 8))

    ax = ax.flatten()

    

    image = cv2.imread(os.path.join(TRAIN_DIR + '/{}.jpg').format(image_data.iloc[0]['image_id']), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)

    

    aug_image = aug_result['image']



    get_bboxes(pascal_voc_boxes, ax[0], color='red')

    orig_bboxes = pascal_voc_boxes

    ax[0].set_title('Original Image with Bounding Boxes')

    ax[0].imshow(image)



    get_bboxes(aug_result['bboxes'], ax[1], color='red')

    aug_bboxes = aug_result['bboxes']

    ax[1].set_title('Augmented Image with Bounding Boxes')

    ax[1].imshow(aug_image)

    

    plt.show()
image = cv2.imread(os.path.join(TRAIN_DIR + '/b6ab77fd7.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)

image_id = 'b6ab77fd7'
pascal_voc_boxes = train_df[train_df['image_id'] == image_id][['x_min', 'y_min', 'x_max', 'y_max']].astype(np.int32).values

labels = np.ones((len(pascal_voc_boxes), ))
aug = A.Compose([

    A.CLAHE(p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.Equalize(p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.Blur(blur_limit=15, p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.RandomCrop(512, 512, p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.Resize(512,512, p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.RandomGamma(p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.ShiftScaleRotate(p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.RandomBrightnessContrast(p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.RandomSizedBBoxSafeCrop(height=512, width = 512, p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.RandomRain(p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.RandomFog(p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.RandomSunFlare(p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])
aug = A.Compose([

    A.ISONoise(p=1)

], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)



show_augmented_images(aug_result, train_df[train_df['image_id'] == 'b6ab77fd7'])