# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from skimage import color
import matplotlib.image as mpimg
import matplotlib.patches as patches
import albumentations as alb
train_path = '/kaggle/input/global-wheat-detection/train.csv'
train_img_path = '/kaggle/input/global-wheat-detection/train'
#read the csv file
train = pd.read_csv(train_path)
train.head()
sns.countplot(train['source'])
#separating x,y,w,h into separate columns for convenience
bboxes = np.stack(train['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep = ',')))
for i, col in enumerate(['x_min', 'y_min', 'w', 'h']):
    train[col] = bboxes[:,i]
#dropping the bbox column as it is not needed now
train.drop(columns = ['bbox'], inplace = True)
#calculate box areas to check for anomaly boxes
train['box_area'] = train['w']*train['h']
#display head of new dataframe
train.head()
#number of unique images in the dataframe
len(train['image_id'].unique())
#number of images in the training directory
len(os.listdir(train_img_path))
#obtaining a list of all images which have no wheat heads in them
unique_imgs_wbox = list(train['image_id'].unique())
all_unique_imgs = os.listdir(train_img_path)
no_wheat_imgs = [img_id for img_id in all_unique_imgs if img_id not in unique_imgs_wbox]
len(no_wheat_imgs)
#append .jpg to image ids for easier handling
train['image_id'] = train['image_id'].apply(lambda x: str(x) + '.jpg')
def get_all_bboxes(df, image_id, count = False):
    '''function that gets all bboxes for a given image id'''
    bboxes = []
    for _,row in df[df.image_id == image_id].iterrows():
        bboxes.append([row.x_min, row.y_min, row.w, row.h])
    if count:
        return bboxes, len(bboxes)
    else:
        return bboxes

def select_img(n, wheat = True):
    '''function to randomly select image ids from the dataframe and return it as a list'''
    if wheat:
        img_ids = train.sample(n = n, random_state = 0)['image_id']
        return list(img_ids)
    else:
        img_ids = np.random.choice(no_wheat_imgs, n)
        return list(img_ids)
        

def plot_imgs(df, ids, bbox = False):
    '''function to plot an even number of images'''
    n = len(ids)
    fig, ax = plt.subplots(2, n//2, figsize = (40,30))
    for i, im_id in enumerate(ids):
        img = mpimg.imread(os.path.join(train_img_path, im_id))
        ax[i//(n//2)][i%(n//2)].imshow(img)
        ax[i//(n//2)][i%(n//2)].axis('off')
        if bbox:
            bboxes = get_all_bboxes(df, im_id)
            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
                ax[i//(n//2)][i%(n//2)].add_patch(rect)
        else:
            pass        
    plt.tight_layout()
    plt.show()
plot_imgs(train, select_img(6))
plot_imgs(train, select_img(6, wheat = False))
plot_imgs(train, select_img(6), bbox = True)
print('Mean box area is: ', train['box_area'].mean())
print('Max box area is: ', train['box_area'].max())
print('Min box area is: ', train['box_area'].min())
#large ids
large_ids = train[train['box_area'] > 170000].image_id
plot_imgs(train, large_ids, bbox = True)
#small ids
small_ids = train[train['box_area']<15].image_id
plot_imgs(train, small_ids, bbox = True)
def get_brightness(image):
    image = color.rgb2gray(image)
    return np.mean(image)*255
#get brightness of each image and append to dataframe
brightness_array = []
image_list = list(train['image_id'].unique())
for img in image_list:
    image = mpimg.imread(os.path.join(train_img_path, img))
    brightness = get_brightness(image)
    brightness_array.append(brightness)

df = pd.DataFrame({'image_id': image_list,
                         'brightness': brightness_array})
df.head()
#bright ids
bright_ids = df[df['brightness'] > 130].image_id
plot_imgs(train, bright_ids[0:6], bbox = True)
#dark ids
dark_ids = df[df['brightness'] < 24].image_id
plot_imgs(train, dark_ids, bbox = True)
print('Mean Brightness is: ', df['brightness'].mean())
print('Max Brightness is: ', df['brightness'].max())
print('Min Brightness is: ', df['brightness'].min())
plt.hist(df['brightness'])
#getting boxes per image
box_count = []
for img in image_list:
    _, count = get_all_bboxes(train, img, count = True)
    box_count.append(count)
    
df['count'] = box_count
df.head()
#more boxes
more_ids = df[df['count'] > 95].image_id
plot_imgs(train, more_ids[0:8], bbox = True)
#less ids 
less_ids = df[df['count']<10].image_id
plot_imgs(train, less_ids[0:8], bbox = True)
print('Mean box count is: ', df['count'].mean())
print('Max box count is: ', df['count'].max())
print('Min box count is: ', df['count'].min())
plt.hist(df['count'])
#describing transforms and the probability of their application 
transforms = alb.Compose([
    alb.HorizontalFlip(p = 0.5),
    alb.VerticalFlip(p = 0.5),
    alb.RandomBrightness(p = 0.2),
    alb.RandomContrast(p = 0.2),
    alb.CLAHE(p = 0.5),
    alb.RandomSizedBBoxSafeCrop(512, 512, erosion_rate = 0.0, interpolation = 1, p = 0.5),
], p=1.0, bbox_params=alb.BboxParams(format='coco', label_fields=['category']))
def apply(transforms, df, n_transforms = 5):
    '''function to apply and view transforms'''
    #randomly choose an image
    img_id = select_img(4) 
    bboxes = get_all_bboxes(df, img_id[3])
    fig,ax = plt.subplots(1, n_transforms + 1, figsize = (40,30))
    image = mpimg.imread(os.path.join(train_img_path, img_id[3]))
    ax[0].imshow(image)
    ax[0].set_title('Original')
    ax[0].axis('off')
    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
        ax[0].add_patch(rect)
    
    #apply transforms one by one and plot
    for i in range(n_transforms):
        parameters = {
            'image': np.asarray(image),
            'bboxes': bboxes,
            'category': np.ones(len(bboxes))
        }
        augmented = transforms(**parameters)
        boxes_aug = augmented['bboxes']
        image_aug = augmented['image']
        ax[i+1].imshow(image_aug)
        ax[i+1].axis('off')
        ax[i+1].set_title('augmented ' + str(i + 1))
        for bbox in boxes_aug:
            rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
            ax[i+1].add_patch(rect)
    plt.tight_layout()
    plt.show()
apply(transforms, train)
apply(transforms, train, 4)