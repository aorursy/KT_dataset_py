# import useful tools
import pandas as pd
import numpy as np
import cv2
import numba 
import ast
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from collections import namedtuple

# import data visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs
from numba import jit
from typing import List, Union, Tuple

# import data augmentation
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

base_path = "/kaggle/input/global-wheat-detection/"
train_dir = f'{base_path}/train/'
test_dir = f'{base_path}/test/'
train_img = glob(train_dir + '*') # create a list of train images
test_img = glob(test_dir + '*') # create a list of test images
#print('Number of train images is {}'.format(len(train_img))) # 3422 train images
#print('Number of test images is {}'.format(len(test_img))) # 10 test images
train_csv_path = f'{base_path}/train.csv'
train = pd.read_csv(train_csv_path) # load dataframe with bboxes
#print(train['bbox'].head(4)) # Show four first rows of the col. bbox


# print(train_img[:4])
# Create a dataframe with all train images
train_images = pd.DataFrame([img.split('/')[-1][:-4] for img in train_img])
train_images.columns=['image_id']

# Merge all train images with the bounding boxes dataframe
train_images = train_images.merge(train, on='image_id', how='left')

# replace nan values with zeros
train_images['bbox'] = train_images.bbox.fillna('[0,0,0,0]')
train_images.head()
"""
train_images = [img.split('/')[-1][:-4] for img in train_img] # removes the .jpg
train_images = [img.split('train/') for img in train_images] # splits the train images into the folder name and img name
train_images = pd.DataFrame(train_images)
train_images.columns = ['image_id']
"""
train_images.drop(columns=['width'], inplace=True) # delete width column
train_images.drop(columns=['height'], inplace=True) # delete height column

# split bbox column
bbox_items = train_images.bbox.str.split(',', expand=True)
train_images['x_min'] = bbox_items[0].str.strip('[ ').astype(float)
train_images['y_min'] = bbox_items[1].str.strip(' ').astype(float)
train_images['width'] = bbox_items[2].str.strip(' ').astype(float)
train_images['height'] = bbox_items[3].str.strip(' ]').astype(float)
train_images.head()
# adding 2 col. x_max and y_max
train_images['x_max'] = train_images.apply(lambda x: x.x_min + x.width, axis=1)
train_images['y_max'] = train_images.apply(lambda y: y.y_min + y.height, axis=1)
# train_images['x_max'] = train_images.apply(lambda x: x.x_min + x.bbox_width, axis=1)
# train_images['y_max'] = train_images.apply(lambda y: y.y_min + y.bbox_height, axis=1)
# train_img.drop(columns=['bbox'], inplace=True) # Delete bbox column

#adding column for bbox_area
train_images['bbox_area'] = train_images.apply(lambda z: z.width * z.height, axis=1)

train_images.head()
"""Create new dataframe with the train images"""


#train_images.drop(columns=['file'], inplace=True)
# train_images = train_images.merge(train, on='image_id', how='left') # merge train images with the bboxes dataframe
# print(train_images.image_id.nunique()) # 3373 train images with bboxes
# print(train_images.shape)
# print(train.shape)
# train_images['bbox'] = train_images.bbox.fillna('[0, 0, 0, 0]') # fill in the nan values with [0, 0, 0, 0]
# bbox_items = train_images.bbox.str.split(',', expand=True) # change the bbox col form [0,0,0,0] to 4 col of 1col: [0
                                                                                            # 2col: 0 3col: 0 4col: 0]
# print('{} images without wheat heads.'.format(len(train_images) - len(train))) #49 images without wheat heads

# Checking bounding box coordinates
# print(max(train_images['x_max'])) # 1024
# print(max(train_images['y_max'])) #1024
# print(min(train_images['x_min'])) #0
# print(min(train_images['y_min'])) #0
x_max = np.array(train_images['x_max'].values.tolist()) # Changing x_max from dataframe to a list
y_max = np.array(train_images['y_max'].values.tolist()) # Changing y_max from dataframe to a list
train_images['x_max'] = np.where(x_max > 1024, 1024, x_max).tolist()
train_images['y_max'] = np.where(y_max > 1024, 1024, y_max).tolist()

len(train_images)
# train_images.loc[train_images.y_max>=1024]
# """image examples"""
# def find_bboxes(df, image_id):
#     img_bbox = df[df['image_id'] == image_id]
#     bboxes = []
#     for col, row in img_bbox.iterrows():
#         bboxes.append((row.x_min, row.y_min, row.width, row.height))
# #         bboxes.append((row.x_min, row.y_min, row.bbox_width, row.bbox_height))
#     return bboxes
# def plt_img(df, rows=3, column=3, title='Image examples'):
#     fig, axs = plt.subplots(rows, column, figsize=(30, 30))
#     for row in range(rows):
#         for col in range(column):
#             idx = np.random.randint(len(df), size=1)[0]
#             img_num = df.iloc[idx].image_id
#             img = Image.open(train_dir + img_num + '.jpg')
#             axs[row, col].imshow(img)
#             bboxes = find_bboxes(df, img_num)

#             for bbox in bboxes:
#                 rect_patch = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='y', facecolor='none')
#                 axs[row, col].add_patch(rect_patch)
#             axs[row, col].axis('off')
#     plt.suptitle(title)

# plt_img(train_images)
"""count the number of bboxes per image"""
train_images['num_bboxes'] = train_images.apply(lambda x: 1 if np.isfinite(x.width) else 0, axis=1)
train_img_count = train_images.groupby('image_id').sum().reset_index() # count the num of bboxes in each image

# def hist_hover(df, column, colors=["#94c8d8", "#ea5e51"], bins=30, title=''):
#     # build histogram data with np
#     hist, edges = np.histogram(df[column], bins=bins)
#     hist_df = pd.DataFrame({column: hist,
#                             "left": edges[:-1],
#                             "right": edges[1:]})
#     hist_df['interval'] = ["%d to %d" % (left, right) for left, right in zip(hist_df['left'], hist_df['right'])]
#     # %d is a numeric or decimalplaceholder
#     # create col. data source in bokeh
#     src = ColumnDataSource(hist_df)
#     plot = figure(plot_height=400, plot_width=600, title=title, x_axis_label=column, y_axis_label='image count')
#     plot.quad(bottom=0, top=column, left='left', right='right', source=src, fill_color=colors[0], line_color='#35838d', fill_alpha=0.7, hover_fill_alpha=0.7, hover_fill_color=colors[1])
#     #hover tool
#     hover = HoverTool(tooltips=[('Interval', '@interval'), ('img count', str("@"+column))])
#     plot.add_tools(hover)
#     #output_file(f'{base_path}/wheat_spikes_per_img.html')
#     output_notebook()
#     show(plot)

# hist_hover(train_img_count, 'num_bboxes', title='Number of wheat spikes per image')
# """# Examples of images with small/large num of spikes
# low_num_of_spikes = train_img_count[train_img_count.num_bboxes < 10].image_id
# plt_img(train_images[train_images.image_id.isin(low_num_of_spikes)], title='Example of images with small number of spikes')
# high_num_of_spikes = train_img_count[train_img_count.num_bboxes > 100].image_id
# plt_img(train_images[train_images.image_id.isin(high_num_of_spikes)], title='Example of images with high number of spikes')
# """
# bbox areas
# train_images['bbox_area'] = train_images.bbox_width * train_images.bbox_height
# hist_hover(train_images, 'bbox_area', title='Area of one bbox')

# print(train_images.bbox_area.max())
# """Because the max area of a bbox is 529788.0 we want to check which image id have big bbox area and delete it
#    Similarly we want to check the min bboxes area and delete it"""
big_bboxes = train_images[train_images.bbox_area > 180000].image_id # 180,000 = 220,000 = 5 images
# plt_img(train_images[train_images.image_id.isin(big_bboxes)], title='Example of images with big bbox area')
# print(big_bboxes)

min_area = train_images[train_images.bbox_area > 0].bbox_area.min()
# print(min_area)
small_bboxes = train_images[(train_images.bbox_area < 50) & (train_images.bbox_area > 0)].image_id # maybe change the 50 number
# plt_img(train_images[train_images.image_id.isin(small_bboxes)], title='Example of images with small bbox area')

"""Area of bounding box per image"""
bbox_area_per_img = train_images.groupby(by='image_id').sum().reset_index()
bbox_percentage = bbox_area_per_img.copy()
bbox_percentage.bbox_area = bbox_percentage.bbox_area/(1024 * 1024) * 100 # normalization of bbox area
# hist_hover(bbox_percentage, 'bbox_area', title='Percentage of image area covered by bboxes')
# print(bbox_percentage.bbox_area.max()) # Max is bigger then 100% (108.19730758666992%) --> bboxes are overlapping


# """Deleting the rows with bbox big or small """
# def bbox_delete(df):
#     for index, col in df.iterrows():
#         if col['bbox_area'] > 180000:
#             df.drop(index, axis=0, inplace=True)
#         # elif (col['bbox_area'] < 50) & (col['bbox_area'] > 0):
#         #     df.drop(index, axis=0, inplace=True)
#         elif (col['x_min'] == 0) & (col['y_min'] == 0) & (col['bbox_width'] == 0) & (col['bbox_height'] == 0):
#             df.drop(index, axis=0, inplace=True)
#         elif (col['x_max'] <= col['x_min']):
#             df.drop(index, axis=0, inplace=True)
#         elif (col['y_max'] <= col['y_min']):
#             df.drop(index, axis=0, inplace=True)
#         elif col['bbox_width'] > 350 or col['bbox_height'] > 350:
#             df.drop(index, axis=0, inplace=True)


def bbox_delete(df):
    for index, col in df.iterrows():
        if col['bbox_area'] > 250000:
            df.drop(index, axis=0, inplace=True)
        # elif (col['bbox_area'] < 50) & (col['bbox_area'] > 0):
        #     df.drop(index, axis=0, inplace=True)
        elif (col['x_min'] == 0) & (col['y_min'] == 0) & (col['width'] == 0) & (col['height'] == 0):
            df.drop(index, axis=0, inplace=True)
        elif (col['x_max'] <= col['x_min']):
            df.drop(index, axis=0, inplace=True)
        elif (col['y_max'] <= col['y_min']):
            df.drop(index, axis=0, inplace=True)
        elif col['width'] > 350 or col['height'] > 350:
            df.drop(index, axis=0, inplace=True)
        

bbox_delete(train_images)
len(train_images)
# hist_hover(train_images, 'bbox_width', title='Histogram of bbox width')
# hist_hover(train_images, 'bbox_height', title='Histogram of bbox height')

# split the train data into train and validation sets (validation set is 15%)
images_ids = train_images.image_id.unique()
train_ids = images_ids[:-510]
valid_ids = images_ids[-510:]

# create dataframes from array
train_df = train_images[train_images.image_id.isin(train_ids)]
valid_df = train_images[train_images.image_id.isin(valid_ids)]
train_df_shape = train_df.shape  # (125689, 13)
valid_df_shape = valid_df.shape  # (22010, 13)
train_df.head()
"""Creating the model"""


class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe.image_id.unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # change the shape from [h,w,c] to [c,h,w]
#         image = torch.from_numpy(image).permute(2,0,1)
        image /= 255.0

        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['x_min', 'y_min', 'width', 'height']].values
#         boxes = records[['x_min', 'y_min', 'bbox_width', 'bbox_height']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        
#         target['boxes'] = torch.zeros((0, 4)) if ('num_bboxes' != 0) else 'bbox' #added to avoid crop issue
        # target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': area, 'iscrowd': iscrowd}
        # target['masks'] = None

        if self.transforms:
            sample = {'image': image, 'bboxes': target['boxes'], 'labels': labels}
            sample = self.transforms(**sample)
            image = sample['image']
#             target['boxes'] = torch.tensor(sample['bboxes']).float()
#             target['boxes'] = torch.tensor(sample['bboxes'])
            target['boxes'] = target['boxes'].type(torch.float32)
            # target['boxes'] = torch.tensor(sample['bboxes']).float()
#             target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0).float()

           
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
# 1 == 1  # checks which device is connected (cpu or gpu)
# # data augmentation visualization for testing

# example_transforms = albu.Compose([
#     albu.RandomResizedCrop(height=256, width=256,scale=(0.08, 1.0), ratio=(1, 1), p=0.5),
#     albu.HorizontalFlip(p=0.5),
#     albu.VerticalFlip(p=0.5),
#     albu.ToSepia(),     
#     albu.OneOf([albu.RGBShift(),
#                 albu.HueSaturationValue(),
#                 albu.RandomGamma(),
#                 albu.RandomBrightness()], p=1.0),
# #     albu.CLAHE(p=0.5)
#     ToTensorV2(p=1.0)
# ], p=1.0, bbox_params=albu.BboxParams(format='coco', min_visibility=0.3, label_fields=['category_id']))

# def apply_transforms(transforms, df, n_transforms=3):
#     idx = np.random.randint(len(df), size=1)[0]
    
#     image_id = df.iloc[idx].image_id
#     bboxes = []
#     for _, row in df[df.image_id == image_id].iterrows():
#         bboxes.append([row.x_min, row.y_min, row.width, row.height])
        
#     image = Image.open(train_dir + image_id + '.jpg')
    
#     fig, axs = plt.subplots(1, n_transforms+1, figsize=(15,7))
    
#     # plot the original image
#     axs[0].imshow(image)
#     axs[0].set_title('original - ' + image_id)
#     for bbox in bboxes:
#         rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
#         axs[0].add_patch(rect)
    
#     # apply transforms n_transforms times
#     for i in range(n_transforms):
#         params = {'image': np.asarray(image),
#                   'bboxes': bboxes,
#                   'category_id': [1 for j in range(len(bboxes))]}
#         augmented_boxes = transforms(**params)
#         bboxes_aug = augmented_boxes['bboxes']
#         image_aug = augmented_boxes['image']

#         # plot the augmented image and augmented bounding boxes
#         axs[i+1].imshow(image_aug)
#         axs[i+1].set_title('augmented_' + str(i+1))
#         for bbox in bboxes_aug:
#             rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
#             axs[i+1].add_patch(rect)
#     plt.show()

# apply_transforms(example_transforms, train_df, n_transforms=3)
# # Comparing without augmentations
# def no_transforms():
#     return albu.Compose([
#         ToTensorV2(p=1.0)
#     ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# def get_valid_transform():
#     return albu.Compose([
#         ToTensorV2(p=1.0)
#     ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# # Albumentations - bbox safe functions for data augmentation
# # took out of crop: erosion_rate=0.0, interpolation=1,
# def train_transforms():
#     return albu.Compose([
# #     albu.RandomResizedCrop(height=256, width=256,scale=(0.08, 1.0), ratio=(1, 1), p=0.5),
#     albu.HorizontalFlip(p=0.5),
#     albu.VerticalFlip(p=0.5),
#     albu.ToSepia(),     
#     albu.OneOf([albu.RGBShift(),
#                 albu.HueSaturationValue(),
#                 albu.RandomGamma(),
#                 albu.RandomBrightness()], p=1.0),
# #     albu.CLAHE(p=0.5),
# #     albu.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
#     ToTensorV2(p=1.0)],
#     p=1.0, bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['labels']))

# def get_valid_transform():
#     return albu.Compose([
#         ToTensorV2(p=1.0)
#     ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))

"""Calculation of IoU"""

@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float: #(0.0 <= IoU <= 1.0)
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()
        
        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]
        
#         Overlap area calculation
    deltax = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    if deltax < 0:
        return 0.0
    
    deltay = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    if deltay < 0:
        return 0.0
    
    area_of_overlap = deltax * deltay
    
    union_area = (((gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)) + ((pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1)) - area_of_overlap)
    
    return area_of_overlap / union_area

        

"""No overlap"""
    
bbox1 = np.array([834.0, 222.0, 56.0, 36.0])
bbox2 = np.array([26.0, 144.0, 124.0, 117.0])
    
assert calculate_iou(bbox1, bbox2, form='coco') == 0

"""Partial overlap"""

bbox1 = np.array([100, 100, 100, 100])
bbox2 = np.array([100, 100, 200, 100])

res = calculate_iou(bbox1, bbox2, form='coco')
assert  res > 0.5 and res < 0.50249

"""Full overlap"""
bbox1 = np.array([834.0, 222.0, 56.0, 36.0])
bbox2 = bbox1

assert calculate_iou(bbox1, bbox2, form='coco') == 1.0
"""Returns the index of the highest IoU between the ground-truth boxes and the prediction"""

@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)"""
    best_match_iou = -np.inf
    best_match_idx = -1
    
    for gt_idx in range(len(gts)):
        if gts[gt_idx][0] < 0: #matches to GT-bbox
            continue 
            
        iou = -1 if ious is None else ious[gt_idx][pred_idx]
        
        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx
"""Calculates precision for GT - prediction pairs at one threshold"""
@jit(nopython=True)
def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    '''    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision'''
    k = len(preds)
    fp = 0 #False positive
    tp = 0 #True positive
    
    
    for prediction_idx in range(k): #pred in enumerate(preds_sorted):

        gt_idx_highest_iou = find_best_match(gts, preds[prediction_idx], prediction_idx,
                                            threshold=threshold, form=form, ious=ious)

        if gt_idx_highest_iou >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[gt_idx_highest_iou] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)
    
'''Calculation of image precision'''

@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds = (0.5, ), form='coco') -> float:
    '''    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision'''
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision
# idx = np.random.randint(len(train_df), size=1)[0]
    
# image_id = train_df.iloc[idx].image_id
# bboxes = []
# for _, row in train_df[train_df.image_id == image_id].iterrows():
#     bboxes.append([row.x_min, row.y_min, row.width, row.height])
           
# image = Image.open(train_dir + image_id + '.jpg')
# transformed = example_transforms(image='image', bboxes='bboxes')
def get_train_transform():
    return albu.Compose([
        albu.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return albu.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


train_dataset = WheatDataset(train_df, train_dir, get_train_transform())
valid_dataset = WheatDataset(valid_df, train_dir, get_valid_transform())
test_dataset = WheatTestDataset(test_df, DIR_TEST, get_test_transform())
# indices = torch.randperm(len(train_dataset)).tolist()#splits dataset into train and val

train_data_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=False, collate_fn=collate_fn)
valid_data_loader = DataLoader(valid_dataset, batch_size=8, num_workers=4, shuffle=True, collate_fn=collate_fn)
test_data_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate_fn)
"""Valid"""
valid_batch = next(iter(valid_data_loader))
images, targets, image_ids = valid_batch
images = list(img.to(device) for img in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
# image_ids = np.array(image_ids)
sample = images[1].permute(1,2,0).cpu().numpy()
# # boxes = targets[2]['boxes'].cpu().numpy().asarray()
# boxes = targets[2]['boxes'].cpu().numpy().astype(np.float32)
# # sample = images[2].cpu().numpy()
# sample = images[2].permute(1,2,0).cpu().numpy()

# fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# for box in boxes:
#     cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2], box[3]),
#                   (220, 0, 0), 3)

# ax.set_axis_off()
# ax.imshow(sample)

"""Train"""
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# lr_scheduler = None
num_epochs = 1

loss_hist = Averager()
# score_hist = Averager()
detection_threshold = 0.5
loss_values = []

itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()
    running_loss = 0.0
#     score_hist.reset()
    
    for images, targets, image_id in train_data_loader:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.train()
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        running_loss =+ loss_value * len(images)
        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step()
      

        
        if itr % 50 == 0:
            print(f"Iteration #{itr} train loss: {loss_value}")
#         loss_values.append(running_loss / len(train_dataset))
        
        itr += 1 

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    print(f"Epoch #{epoch} Loss: {loss_hist.value}")
    
    for i, data in enumerate(train_data_loader, 0):
        running_loss =+ loss_value * len(images)
    loss_values.append(running_loss / len(train_dataset))

    with torch.no_grad():

        validation_image_precisions = []
        iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]

        for images, targets, image_ids in valid_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            model.eval()
            outputs = model(images, targets)

#         outputs = model.forward(images)
        # validation losses

        # Calculate validation losses
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         loss_value = losses.item()


        # validation score

            # Calculating mAP@
            for i, image in enumerate(images):
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].data.cpu().numpy()
#                 boxes = boxes[scores >= detection_threshold].astype(np.int32)
                target = targets[i]['boxes'].cpu().data.numpy()
    #             sort_pred_idx = np.argsort(scores)[::-1]
    #             sort_pred = boxes[sort_pred_idx]
                    # shape is x1,y1,x2,y2 (pascal_voc)
                image_precision = calculate_image_precision(target,
                                                        boxes,
                                                        thresholds=iou_thresholds,
                                                        form='pascal_voc')

                validation_image_precisions.append(image_precision)
    #         if itr % 50 == 0:
    #             print(f"Iteration #{itr} loss: {loss_value}")
    #             print(f"Iteration #{itr} train loss: {loss_value}")
    #             print(f"Iteration #{itr} score: {np.mean(validation_image_precisions)}")

    #         itr += 1

    print(f"Validation #{epoch} score: {np.mean(validation_image_precisions)}")

    print("Validation IOU: {0:.4f}".format(np.mean(validation_image_precisions)))
plt.plot(loss_values)

"""If we need to calculate validation loss (i think we don't need to cause we don't train the validation??)
#             model.train()
#                 # Calculate validation losses
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
#             loss_value = losses.item()
##             output = model(images, targets)
# #            loss = criterion(output, target)
              running_loss =+ loss_value * len(images)
              loss_hist.send(loss_value)

              optimizer.zero_grad()
              losses.backward()
              optimizer.step()
              lr_scheduler.step()

#             if itr % 50 == 0:
#                 print(f"Validation loss #{epoch} score: {np.mean(validation_image_precisions)}")

#             itr += 1 

#         print(f"Validation loss #{epoch} score: {np.mean(validation_image_precisions)}")
"""
loss_values
loss_hist
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
# Our testing sample
sample_id = '1ef16dab1'

gt_boxes = train_df[train_df['image_id'] == sample_id][['x_min', 'y_min', 'width', 'height']].values
gt_boxes = gt_boxes.astype(np.int)

# Ground-truth boxes of our sample
gt_boxes
# No GT - Predicted box match
pred_box = np.array([0, 0, 10, 10])
assert find_best_match(gt_boxes, pred_box, 0, threshold=0.5, form='coco') == -1

# First GT match
pred_box = np.array([954., 391., 70., 90.])
assert find_best_match(gt_boxes, pred_box, 0, threshold=0.5, form='coco') == 0

# These are the predicted boxes (and scores) from my locally trained model.
preds = np.array([[956, 409, 68, 85],
                  [883, 945, 85, 77],
                  [745, 468, 81, 87],
                  [658, 239, 103, 105],
                  [518, 419, 91, 100],
                  [711, 805, 92, 106],
                  [62, 213, 72, 64],
                  [884, 175, 109, 68],
                  [721, 626, 96, 104],
                  [878, 619, 121, 81],
                  [887, 107, 111, 71],
                  [827, 525, 88, 83],
                  [816, 868, 102, 86],
                  [166, 882, 78, 75],
                  [603, 563, 78, 97],
                  [744, 916, 68, 52],
                  [582, 86, 86, 72],
                  [79, 715, 91, 101],
                  [246, 586, 95, 80],
                  [181, 512, 93, 89],
                  [655, 527, 99, 90],
                  [568, 363, 61, 76],
                  [9, 717, 152, 110],
                  [576, 698, 75, 78],
                  [805, 974, 75, 50],
                  [10, 15, 78, 64],
                  [826, 40, 69, 74],
                  [32, 983, 106, 40]]
                )

scores = np.array([0.9932319, 0.99206185, 0.99145633, 0.9898089, 0.98906296, 0.9817738,
                   0.9799762, 0.97967803, 0.9771589, 0.97688967, 0.9562935, 0.9423076,
                   0.93556845, 0.9236257, 0.9102379, 0.88644403, 0.8808225, 0.85238415,
                   0.8472188, 0.8417798, 0.79908705, 0.7963756, 0.7437897, 0.6044758,
                   0.59249884, 0.5557045, 0.53130984, 0.5020239])


# Sort highest confidence -> lowest confidence
preds_sorted_idx = np.argsort(scores)[::-1]
preds_sorted = preds[preds_sorted_idx]

def show_result(sample_id, preds, gt_boxes):
    sample = cv2.imread(f'{train_dir}/{sample_id}.jpg', cv2.IMREAD_COLOR)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for pred_box in preds:
        cv2.rectangle(
            sample,
            (pred_box[0], pred_box[1]),
            (pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]),
            (220, 0, 0), 2
        )

    for gt_box in gt_boxes:    
        cv2.rectangle(
            sample,
            (gt_box[0], gt_box[1]),
            (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]),
            (0, 220, 0), 2
        )
# cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# ax.putText(image, "IoU: {:.4f}".format(iou), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    ax.set_axis_off()
    ax.imshow(sample)
    ax.set_title("RED: Predicted | GREEN - Ground-truth")
show_result(sample_id, preds, gt_boxes)
# from visdom import Visdom

# viz = Visdom()
# # create and initialize
# viz.line([[0., 0.]], [0], win='train', opts=dict(title='loss&amp;acc', legend=['loss', 'acc']))

# for global_steps in range(10):

#     train_loss = loss_hist.value
#     train_acc = accuracy
#     # just for example
# #     train_loss = 0.1 * np.random.randn() + 1
# #     train_acc = 0.1 * np.random.randn() + 0.5

#     # update the window
#     viz.line([[train_loss, train_acc]], [global_steps], win='train', update='append')

#     time.sleep(0.5)

# val_img_precisions = []
# thresh_iou = [x for x in np.arange(0.5, 0.76, 0.05)]
# for images, targets, image_ids in valid_data_loader:
#     gt_boxes = 
#     sort_pred_idx = np.argsort(scores)[::-1]
#     sort_pred = target[sort_pred_idx]
    
#     for idx, img in enumerate(images):
#         img_precision = calculate_image_precision(sort_pred, boxes, thresholds=iou_thresholds, form='coco')
#         valid_img_precisions.append(img_preciosion)
    
# print("Validation IOU: {0:.4f}".format(np.mean(validation_image_precisions)))
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)

ax.set_axis_off()
ax.imshow(sample)

# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.eval()

x = model.to(device)
def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
detection_threshold = 0.5
results = []

for images, image_ids in test_data_loader:

    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }

        
        results.append(result)

results[0:2]
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head()
sample = images[1].permute(1,2,0).cpu().numpy()
boxes = outputs[1]['boxes'].data.cpu().numpy()
scores = outputs[1]['scores'].data.cpu().numpy()

boxes = boxes[scores >= detection_threshold].astype(np.int32)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 2)
    
ax.set_axis_off()
ax.imshow(sample)
test_df.to_csv('submission.csv', index=False)
