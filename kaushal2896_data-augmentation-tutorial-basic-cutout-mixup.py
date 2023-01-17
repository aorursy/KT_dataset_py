import os

import ast

from collections import namedtuple



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tqdm import tqdm

from PIL import Image



import joblib

from joblib import Parallel, delayed



import cv2

import albumentations

from albumentations.pytorch.transforms import ToTensorV2

from albumentations.core.transforms_interface import DualTransform

from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox



from sklearn.model_selection import StratifiedKFold



import torch

from torch.utils.data import DataLoader, Dataset

import torch.utils.data as data_utils



from matplotlib import pyplot as plt

import matplotlib.patches as patches

from matplotlib.image import imsave
# Constants

BASE_DIR = '/kaggle/input/global-wheat-detection'

WORK_DIR = '/kaggle/working'

BATCH_SIZE = 16



# Set seed for numpy for reproducibility

np.random.seed(1996)
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))



# Let's expand the bounding box coordinates and calculate the area of all the bboxes

train_df[['x_min','y_min', 'width', 'height']] = pd.DataFrame([ast.literal_eval(x) for x in train_df.bbox.tolist()], index= train_df.index)

train_df = train_df[['image_id', 'bbox', 'source', 'x_min', 'y_min', 'width', 'height']]

train_df['area'] = train_df['width'] * train_df['height']

train_df['x_max'] = train_df['x_min'] + train_df['width']

train_df['y_max'] = train_df['y_min'] + train_df['height']

train_df = train_df.drop(['bbox'], axis=1)

train_df = train_df[['image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height', 'area', 'source']]



# There are some buggy annonations in training images having huge bounding boxes. Let's remove those bboxes

train_df = train_df[train_df['area'] < 100000]



train_df.head()
print(train_df.shape)
image_ids = train_df['image_id'].unique()

print(f'Total number of training images: {len(image_ids)}')
# Read the image on which data augmentaion is to be performed

image_id = 'c14c1e300'

image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image /= 255.0

plt.figure(figsize = (10, 10))

plt.imshow(image)

plt.show()
pascal_voc_boxes = train_df[train_df['image_id'] == image_id][['x_min', 'y_min', 'x_max', 'y_max']].astype(np.int32).values

coco_boxes = train_df[train_df['image_id'] == image_id][['x_min', 'y_min', 'width', 'height']].astype(np.int32).values

assert(len(pascal_voc_boxes) == len(coco_boxes))

labels = np.ones((len(pascal_voc_boxes), ))
def get_bbox(bboxes, col, color='white', bbox_format='pascal_voc'):

    

    for i in range(len(bboxes)):

        # Create a Rectangle patch

        if bbox_format == 'pascal_voc':

            rect = patches.Rectangle(

                (bboxes[i][0], bboxes[i][1]),

                bboxes[i][2] - bboxes[i][0], 

                bboxes[i][3] - bboxes[i][1], 

                linewidth=2, 

                edgecolor=color, 

                facecolor='none')

        else:

            rect = patches.Rectangle(

                (bboxes[i][0], bboxes[i][1]),

                bboxes[i][2], 

                bboxes[i][3], 

                linewidth=2, 

                edgecolor=color, 

                facecolor='none')



        # Add the patch to the Axes

        col.add_patch(rect)
aug = albumentations.Compose([

        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512

        albumentations.VerticalFlip(1),    # Verticlly flip the image

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=pascal_voc_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(pascal_voc_boxes, ax[0], color='red')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red')

ax[1].title.set_text('Augmented Image')

ax[1].imshow(aug_result['image'])

plt.show()
aug = albumentations.Compose([

        albumentations.Resize(512, 512),   # Resize the given 1024 x 1024 image to 512 * 512

        albumentations.VerticalFlip(1),     # Verticlly flip the image

        albumentations.Blur(p=1)

    ], bbox_params={'format': 'coco', 'label_fields': ['labels']})
aug_result = aug(image=image, bboxes=coco_boxes, labels=labels)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

get_bbox(coco_boxes, ax[0], color='red', bbox_format='coco')

ax[0].title.set_text('Original Image')

ax[0].imshow(image)



get_bbox(aug_result['bboxes'], ax[1], color='red', bbox_format='coco')

ax[1].title.set_text('Augmented Image')

ax[1].imshow(aug_result['image'])

plt.show()
class CustomCutout(DualTransform):

    """

    Custom Cutout augmentation with handling of bounding boxes 

    Note: (only supports square cutout regions)

    

    Author: Kaushal28

    Reference: https://arxiv.org/pdf/1708.04552.pdf

    """

    

    def __init__(

        self,

        fill_value=0,

        bbox_removal_threshold=0.50,

        min_cutout_size=192,

        max_cutout_size=512,

        always_apply=False,

        p=0.5

    ):

        """

        Class construstor

        

        :param fill_value: Value to be filled in cutout (default is 0 or black color)

        :param bbox_removal_threshold: Bboxes having content cut by cutout path more than this threshold will be removed

        :param min_cutout_size: minimum size of cutout (192 x 192)

        :param max_cutout_size: maximum size of cutout (512 x 512)

        """

        super(CustomCutout, self).__init__(always_apply, p)  # Initialize parent class

        self.fill_value = fill_value

        self.bbox_removal_threshold = bbox_removal_threshold

        self.min_cutout_size = min_cutout_size

        self.max_cutout_size = max_cutout_size

        

    def _get_cutout_position(self, img_height, img_width, cutout_size):

        """

        Randomly generates cutout position as a named tuple

        

        :param img_height: height of the original image

        :param img_width: width of the original image

        :param cutout_size: size of the cutout patch (square)

        :returns position of cutout patch as a named tuple

        """

        position = namedtuple('Point', 'x y')

        return position(

            np.random.randint(0, img_width - cutout_size + 1),

            np.random.randint(0, img_height - cutout_size + 1)

        )

        

    def _get_cutout(self, img_height, img_width):

        """

        Creates a cutout pacth with given fill value and determines the position in the original image

        

        :param img_height: height of the original image

        :param img_width: width of the original image

        :returns (cutout patch, cutout size, cutout position)

        """

        cutout_size = np.random.randint(self.min_cutout_size, self.max_cutout_size + 1)

        cutout_position = self._get_cutout_position(img_height, img_width, cutout_size)

        return np.full((cutout_size, cutout_size, 3), self.fill_value), cutout_size, cutout_position

        

    def apply(self, image, **params):

        """

        Applies the cutout augmentation on the given image

        

        :param image: The image to be augmented

        :returns augmented image

        """

        image = image.copy()  # Don't change the original image

        self.img_height, self.img_width, _ = image.shape

        cutout_arr, cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width)

        

        # Set to instance variables to use this later

        self.image = image

        self.cutout_pos = cutout_pos

        self.cutout_size = cutout_size

        

        image[cutout_pos.y:cutout_pos.y+cutout_size, cutout_pos.x:cutout_size+cutout_pos.x, :] = cutout_arr

        return image

    

    def apply_to_bbox(self, bbox, **params):

        """

        Removes the bounding boxes which are covered by the applied cutout

        

        :param bbox: A single bounding box coordinates in pascal_voc format

        :returns transformed bbox's coordinates

        """



        # Denormalize the bbox coordinates

        bbox = denormalize_bbox(bbox, self.img_height, self.img_width)

        x_min, y_min, x_max, y_max = tuple(map(int, bbox))



        bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height

        overlapping_size = np.sum(

            (self.image[y_min:y_max, x_min:x_max, 0] == self.fill_value) &

            (self.image[y_min:y_max, x_min:x_max, 1] == self.fill_value) &

            (self.image[y_min:y_max, x_min:x_max, 2] == self.fill_value)

        )



        # Remove the bbox if it has more than some threshold of content is inside the cutout patch

        if overlapping_size / bbox_size > self.bbox_removal_threshold:

            return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)



        return normalize_bbox(bbox, self.img_height, self.img_width)



    def get_transform_init_args_names(self):

        """

        Fetches the parameter(s) of __init__ method

        :returns: tuple of parameter(s) of __init__ method

        """

        return ('fill_value', 'bbox_removal_threshold', 'min_cutout_size', 'max_cutout_size', 'always_apply', 'p')
augmentation = albumentations.Compose([

    CustomCutout(p=1),

    albumentations.Flip(always_apply=True), # Either Horizontal, Vertical or both flips

    albumentations.OneOf([  # One of blur or adding gauss noise

        albumentations.Blur(p=0.50),  # Blurs the image

        albumentations.GaussNoise(var_limit=5.0 / 255.0, p=0.50)  # Adds Gauss noise to image

    ], p=1)

], bbox_params = {

    'format': 'pascal_voc',

    'label_fields': ['labels']

})
def get_bbox(bboxes, col, color='white'):

    for i in range(len(bboxes)):

        # Create a Rectangle patch

        rect = patches.Rectangle(

            (bboxes[i][0], bboxes[i][1]),

            bboxes[i][2] - bboxes[i][0], 

            bboxes[i][3] - bboxes[i][1], 

            linewidth=2, 

            edgecolor=color, 

            facecolor='none')



        # Add the patch to the Axes

        col.add_patch(rect)
num_images = 5

rand_start = np.random.randint(0, len(image_ids) - 5)

fig, ax = plt.subplots(nrows=num_images, ncols=2, figsize=(16, 40))



for index, image_id in enumerate(image_ids[rand_start : rand_start + num_images]):

    # Read the image from image id

    image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    image /= 255.0  # Normalize

    

    # Get the bboxes details and apply all the augmentations

    bboxes = train_df[train_df['image_id'] == image_id][['x_min', 'y_min', 'x_max', 'y_max']].astype(np.int32).values

    labels = np.ones((len(bboxes), ))  # As we have only one class (wheat heads)

    aug_result = augmentation(image=image, bboxes=bboxes, labels=labels)



    get_bbox(bboxes, ax[index][0], color='red')

    ax[index][0].grid(False)

    ax[index][0].set_xticks([])

    ax[index][0].set_yticks([])

    ax[index][0].title.set_text('Original Image')

    ax[index][0].imshow(image)



    get_bbox(aug_result['bboxes'], ax[index][1], color='red')

    ax[index][1].grid(False)

    ax[index][1].set_xticks([])

    ax[index][1].set_yticks([])

    ax[index][1].title.set_text(f'Augmented Image: Removed bboxes: {len(bboxes) - len(aug_result["bboxes"])}')

    ax[index][1].imshow(aug_result['image'])

plt.show()
def mixup(images, bboxes, areas, alpha=1.0):

    """

    Randomly mixes the given list if images with each other

    

    :param images: The images to be mixed up

    :param bboxes: The bounding boxes (labels)

    :param areas: The list of area of all the bboxes

    :param alpha: Required to generate image wieghts (lambda) using beta distribution. In this case we'll use alpha=1, which is same as uniform distribution

    """

    # Generate random indices to shuffle the images

    indices = torch.randperm(len(images))

    shuffled_images = images[indices]

    shuffled_bboxes = bboxes[indices]

    shuffled_areas = areas[indices]

    

    # Generate image weight (minimum 0.4 and maximum 0.6)

    lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)

    print(f'lambda: {lam}')

    

    # Weighted Mixup

    mixedup_images = lam*images + (1 - lam)*shuffled_images

    

    mixedup_bboxes, mixedup_areas = [], []

    for bbox, s_bbox, area, s_area in zip(bboxes, shuffled_bboxes, areas, shuffled_areas):

        mixedup_bboxes.append(bbox + s_bbox)

        mixedup_areas.append(area + s_area)

    

    return mixedup_images, mixedup_bboxes, mixedup_areas, indices.numpy()
class WheatDataset(Dataset):

    

    def __init__(self, df):

        self.df = df

        self.image_ids = self.df['image_id'].unique()



    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0  # Normalize

        

        # Get bbox coordinates for each wheat head(s)

        bboxes_df = self.df[self.df['image_id'] == image_id]

        boxes, areas = [], []

        n_objects = len(bboxes_df)  # Number of wheat heads in the given image



        for i in range(n_objects):

            x_min = bboxes_df.iloc[i]['x_min']

            x_max = bboxes_df.iloc[i]['x_max']

            y_min = bboxes_df.iloc[i]['y_min']

            y_max = bboxes_df.iloc[i]['y_max']



            boxes.append([x_min, y_min, x_max, y_max])

            areas.append(bboxes_df.iloc[i]['area'])



        return {

            'image_id': image_id,

            'image': image,

            'boxes': boxes,

            'area': areas,

        }
def collate_fn(batch):

    images, bboxes, areas, image_ids = ([] for _ in range(4))

    for data in batch:

        images.append(data['image'])

        bboxes.append(data['boxes'])

        areas.append(data['area'])

        image_ids.append(data['image_id'])



    return np.array(images), np.array(bboxes), np.array(areas), np.array(image_ids)     
train_dataset = WheatDataset(train_df)

train_loader = data_utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
images, bboxes, areas, image_ids = next(iter(train_loader))

aug_images, aug_bboxes, aug_areas, aug_indices = mixup(images, bboxes, areas)
def read_image(image_id):

    """Read the image from image id"""



    image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    image /= 255.0  # Normalize

    return image
fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(15, 30))

for index in range(5):

    image_id = image_ids[index]

    image = read_image(image_id)



    get_bbox(bboxes[index], ax[index][0], color='red')

    ax[index][0].grid(False)

    ax[index][0].set_xticks([])

    ax[index][0].set_yticks([])

    ax[index][0].title.set_text('Original Image #1')

    ax[index][0].imshow(image)

    

    image_id = image_ids[aug_indices[index]]

    image = read_image(image_id)

    get_bbox(bboxes[aug_indices[index]], ax[index][1], color='red')

    ax[index][1].grid(False)

    ax[index][1].set_xticks([])

    ax[index][1].set_yticks([])

    ax[index][1].title.set_text('Original Image #2')

    ax[index][1].imshow(image)



    get_bbox(aug_bboxes[index], ax[index][2], color='red')

    ax[index][2].grid(False)

    ax[index][2].set_xticks([])

    ax[index][2].set_yticks([])

    ax[index][2].title.set_text(f'Augmented Image: lambda * image1 + (1 - lambda) * image2')

    ax[index][2].imshow(aug_images[index])

plt.show()
augmentation = albumentations.Compose([

    albumentations.Flip(p=0.60),

    albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.60),

    albumentations.HueSaturationValue(p=0.60)

], bbox_params = {

    'format': 'pascal_voc',

    'label_fields': ['labels']

})
if not os.path.isdir('train'):

    os.mkdir('train')
def create_dataset(index, image_id):

    # Read the image from image id

    image = cv2.imread(os.path.join(BASE_DIR, 'train', f'{image_id}.jpg'), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    # Get the bboxes details and apply all the augmentations

    bboxes = train_df[train_df['image_id'] == image_id][['x_min', 'y_min', 'x_max', 'y_max']].astype(np.int32).values

    source = train_df[train_df['image_id'] == image_id]['source'].unique()[0]

    labels = np.ones((len(bboxes), ))  # As we have only one class (wheat heads)

    aug_result = augmentation(image=image, bboxes=bboxes, labels=labels)



    aug_image = aug_result['image']

    aug_bboxes = aug_result['bboxes']

    

    Image.fromarray(image).save(os.path.join(WORK_DIR, 'train', f'{image_id}.jpg'))

    Image.fromarray(aug_image).save(os.path.join(WORK_DIR, 'train', f'{image_id}_aug.jpg'))



    image_metadata = []

    for bbox in aug_bboxes:

        bbox = tuple(map(int, bbox))

        image_metadata.append({

            'image_id': f'{image_id}_aug',

            'x_min': bbox[0],

            'y_min': bbox[1],

            'x_max': bbox[2],

            'y_max': bbox[3],

            'width': bbox[2] - bbox[0],

            'height': bbox[3] - bbox[1],

            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),

            'source': source

        })

    return image_metadata
image_metadata = Parallel(n_jobs=8)(delayed(create_dataset)(index, image_id) for index, image_id in tqdm(enumerate(image_ids), total=len(image_ids)))

image_metadata = [item for sublist in image_metadata for item in sublist]
aug_train_df = pd.DataFrame(image_metadata)

aug_train_df.head()
print(aug_train_df.shape)
train_df = pd.concat([train_df, aug_train_df]).reset_index(drop=True)

train_df
train_df.shape
# Add a new column to store kfold indices

train_df.loc[:, 'kfold'] = -1
image_source = train_df[['image_id', 'source']].drop_duplicates()



# get lists for image_ids and sources

image_ids = image_source['image_id'].to_numpy()

sources = image_source['source'].to_numpy()



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1996)

split = skf.split(image_ids, sources) # second arguement is what we are stratifying by



for fold, (train_idx, val_idx) in enumerate(split):

    translated_val_idx = train_df[train_df['image_id'].isin(image_ids[val_idx])].index.values

    print(len(translated_val_idx))

    train_df.loc[translated_val_idx, 'kfold'] = fold
train_df
train_df.to_csv('train.csv', index=False)
!zip -r -qq train.zip train

!rm -rf train