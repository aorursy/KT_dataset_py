## forked from https://www.kaggle.com/titericz/building-and-visualizing-masks



%matplotlib inline

import matplotlib.pyplot as plt

import random

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

train_masks = train.loc[

    lambda df: df['EncodedPixels'].notnull()

].assign(

    image_id=lambda df: df['ImageId_ClassId'].str.split('_', expand=True)[0],

    class_id=lambda df: df['ImageId_ClassId'].str.split('_', expand=True)[1],

)

train_image_ids = train_masks.loc[:,'image_id'].drop_duplicates()

print('Number of RLEs: ', len(train_masks))

print('Number of train images with RLEs: ', len(train_image_ids))

train_masks.head(10)
train_mask_counts = train_masks.groupby('image_id').agg({'class_id': 'count'}).reset_index()

dist_non_null_train_masks = train_mask_counts.groupby('class_id').size()

dist_non_null_train_masks.head()

three_class_images = train_mask_counts.loc[lambda df: df.class_id == 3, 'image_id']
def rle2mask(rle, imgshape):

    width = imgshape[0]

    height= imgshape[1]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mask.reshape(height,width), k=1 ) )
def filter_images_w_classes(class_id):

    image_ids = train_masks.loc[lambda df: df.class_id == str(class_id), 'image_id'].drop_duplicates()

    return image_ids



class_image_ids = {    

    i: filter_images_w_classes(i)

    for i in range(1, 5)

}
def plot_images(class_id):

    fig=plt.figure(figsize=(20,150))

    images = 8

    columns = 4

    horizontal_splits = 6

    rows = int(images * horizontal_splits * 2 / columns)



    rgb_class_ids = {

        '1': (255, 0, 0),

        '2': (0, 255, 0),

        '3': (0, 0, 255),

        '4': (255, 150, 0),

    }

    for i_image, image_id in enumerate(class_image_ids[class_id].sample(images).values):



        img = cv2.imread( '../input/train_images/' + image_id)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_w_masks = img.copy()



        chunk_x_span = int(img.shape[1] / horizontal_splits)



        classed_masks = train_masks.loc[lambda df: df.image_id == image_id]

        for class_id, rle in classed_masks.loc[:,['class_id', 'EncodedPixels']].values:

            mask = rle2mask(rle, img.shape)

            img_w_masks[mask==1] = rgb_class_ids[class_id]



        for i_chunk in range(horizontal_splits):

            index_no_mask = i_image*horizontal_splits*2 + i_chunk*2 + 1

            index_with_mask = index_no_mask + 1

            fig.add_subplot(rows, columns, index_no_mask, title='{} [Chunk {}]'.format(image_id, i_chunk))

            plt.imshow(img[:, chunk_x_span*i_chunk:chunk_x_span*(i_chunk + 1)])

            fig.add_subplot(rows, columns, index_with_mask, title='{} [Chunk {}] with masks'.format(image_id, i_chunk))

            plt.imshow(img_w_masks[:, chunk_x_span*i_chunk:chunk_x_span*(i_chunk + 1)])

    plt.show()
plot_images(1)
plot_images(2)
plot_images(3)
plot_images(4)
fig=plt.figure(figsize=(20,50))

images = 2

columns = 4

horizontal_splits = 6

rows = images * horizontal_splits



rgb_class_ids = {

    '1': (255, 0, 0),

    '2': (0, 255, 0),

    '3': (0, 0, 255),

    '4': (255, 255, 255),

}

for i_image, image_id in enumerate(three_class_images[:images].values):

    

    img = cv2.imread( '../input/train_images/' + image_id)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_w_masks = img.copy()

    

    chunk_x_span = int(img.shape[1] / horizontal_splits)

    

    classed_masks = train_masks.loc[lambda df: df.image_id == image_id]

    for class_id, rle in classed_masks.loc[:,['class_id', 'EncodedPixels']].values:

        mask = rle2mask(rle, img.shape)

        img_w_masks[mask==1] = rgb_class_ids[class_id]

    

    for i_chunk in range(horizontal_splits):

        index_no_mask = i_image*horizontal_splits*2 + i_chunk*2 + 1

        index_with_mask = i_image*horizontal_splits*2 + i_chunk*2 + 2

        fig.add_subplot(rows, columns, index_no_mask, title='{} [Chunk {}]'.format(image_id, i_chunk))

        plt.imshow(img[:, chunk_x_span*i_chunk:chunk_x_span*(i_chunk + 1)])

        fig.add_subplot(rows, columns, index_with_mask, title='{} [Chunk {}] with masks'.format(image_id, i_chunk))

        plt.imshow(img_w_masks[:, chunk_x_span*i_chunk:chunk_x_span*(i_chunk + 1)])

plt.show()