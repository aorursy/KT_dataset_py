import os
from ast import literal_eval

import numpy as np
import pandas as pd

import cv2

import matplotlib.pyplot as plt
train_img_path = '../input/global-wheat-detection/train'

train_files = os.listdir(train_img_path)
train_list = [x[:-4] for x in train_files]

train_ids = pd.DataFrame(train_list, columns=['image_id'])
train_ids = train_ids.set_index('image_id')

train_ids_sample = train_ids.sample(n=25)

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axs.flat):
    image_id = train_ids_sample.iloc[i].name
    image_bgr = cv2.imread("%s/%s.jpg" % (train_img_path, image_id))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    ax.set_title(image_id)
test_img_path = '../input/global-wheat-detection/test'

test_files = os.listdir(test_img_path)
test_list = [x[:-4] for x in test_files]

test_ids = pd.DataFrame(test_list, columns=['image_id'])
test_ids = test_ids.set_index('image_id')

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axs.flat):
    image_id = test_ids.iloc[i].name
    image_bgr = cv2.imread("%s/%s.jpg" % (test_img_path, image_id))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    ax.set_title(image_id)
train_meta = pd.read_csv('../input/global-wheat-detection/train.csv',
                    converters={'bbox': literal_eval})
sample_submission = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')

print(train_meta)
print(sample_submission)
# Get info about images from train.csv
train_info = train_meta.groupby('image_id').agg({
    'width': np.min, 'height': np.min, 'bbox': np.size})
train_info = train_info.rename(columns={'bbox': 'bbox_no'})

# Join image info to our original list of images
train = pd.merge(train_ids, train_info, on='image_id', how='left')
print(train.info())

train_no_info = train[train['bbox_no'].isnull()]
print(train_no_info.info())
train_no_info_sample = train_no_info.sample(n=25)

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axs.flat):
    image_id = train_no_info_sample.iloc[i].name
    image_bgr = cv2.imread("%s/%s.jpg" % (train_img_path, image_id))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    ax.set_title(image_id)
train[train['bbox_no'].isnull()] = 0

fig, ax = plt.subplots()
_ = ax.hist(train['bbox_no'], bins=50)
_ = ax.set_xlabel('No. of bounding boxes')
_ = ax.set_ylabel('Frequency')
_ = ax.set_title('Histogram of bounding boxes per image')
train_sample = train.sample(n=25)

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axs.flat):
    image_id = train_sample.iloc[i].name

    image_bgr = cv2.imread("%s/%s.jpg" % (train_img_path, image_id))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    bboxes = train_meta[train_meta['image_id'] == image_id]['bbox']    
    
    ax.imshow(image)
    ax.set_title(image_id)
    
    # draw rectangle for each bounding box
    for bbox in bboxes:
        [xmin, ymin, width, height] = bbox
        ax.add_patch(
            plt.Rectangle((xmin, ymin), width, height,
                          fill=False, edgecolor='r', linewidth=2, alpha=0.5)
        )
train_meta['box_width'] = train_meta['bbox'].apply(lambda x: x[2])
train_meta['box_height'] = train_meta['bbox'].apply(lambda x: x[3])

fig, ax = plt.subplots()
_ = ax.scatter(train_meta['box_width'], train_meta['box_height'])
_ = ax.set_xlabel('Bounding box width (pixels)')
_ = ax.set_ylabel('Bounding box height (pixels)')
_ = ax.set_title('Scatter plot of bounding box width and heigth')
train_meta_outlier_box = train_meta[
                                    (train_meta['box_width'] < 10)
                                    | (train_meta['box_height'] < 10)
                                    | (train_meta['box_width'] > 400)
                                    | (train_meta['box_height'] > 400)
                                  ]

train_outlier_box = train_meta_outlier_box[['image_id', 'bbox']].groupby('image_id').agg({
    'bbox': np.size})
train_outlier_box = train_outlier_box.rename(columns={'bbox': 'bbox_no'})
# train_outlier_box_sample = train_outlier_box.sample(n=25)

train_outlier_box_sample = train[:4]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axs.flat):
    image_id = train_outlier_box_sample.iloc[i].name

    image_bgr = cv2.imread("%s/%s.jpg" % (train_img_path, image_id))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     bboxes = train_meta_outlier_box[train_meta_outlier_box['image_id'] == image_id]['bbox']    
    bboxes = train_meta[train_meta['image_id'] == image_id]['bbox']        
    
    ax.imshow(image)
    ax.set_title(image_id)
    
    # draw rectangle for each bounding box
    for bbox in bboxes:
        [xmin, ymin, width, height] = bbox
        ax.add_patch(
            plt.Rectangle((xmin, ymin), width, height,
                          fill=False, edgecolor='red', linewidth=5, alpha=1)
        )