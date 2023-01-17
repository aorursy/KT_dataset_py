import numpy as np
import pandas as pd
import os
import glob
import cv2
import math
import seaborn as sns
import json

sns.set()
sns.set_palette("husl")

TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

RANDOM_SEED=23
OUTPUT_PATH = './'
train_ids = [x for x in os.listdir(TRAIN_PATH)]
test_ids = [x for x in os.listdir(TEST_PATH)]
df = pd.DataFrame({'id':train_ids,'train_or_test':'train'})
df = df.append(pd.DataFrame({'id':test_ids,'train_or_test':'test'}))
df.groupby(['train_or_test']).count()
#Building the paths for individual images:
df['path'] = df.apply(lambda x:'../input/stage1_{}/{}/images/{}.png'.format(x[1],x[0],x[0]), axis=1)
#Storing the shapes of the images in the dataframe:
from scipy import misc

df['shape'] = ''
for index, row in df.iterrows():
    image = misc.imread(row['path'])
    df.set_value(index, 'shape', str(image.shape))
%matplotlib inline
from matplotlib import pyplot as plt

agg = df[['shape','train_or_test','id']].groupby(['shape','train_or_test']).count().unstack()
agg.columns = agg.columns.droplevel()

agg.plot.barh(stacked=True,figsize=(16,4))
plt.show()
def n_of_each(df, n = 4):
    shapes = df['shape'].unique()
    sample = pd.DataFrame()
    
    for shape in shapes:
        sample = sample.append(df[df['shape']==shape].sample(n, replace=True))
    
    return sample.sort_values(by=['shape']).reset_index()

def show_image(ax,title,image):
    ax.grid(None)
    ax.set_title(title)
    ax.imshow(image)
    
def show_row_col(sample,cols,path_col='path',image_col=None,label_col='title',mode='file'):
    rows = math.ceil(len(sample)/cols)
    
    fig, ax = plt.subplots(rows,cols,figsize=(5*cols,5*rows))
    
    for index, data in sample.iterrows():
    
        title = data[label_col]
        if mode=='image':
            image = np.array(data[image_col],dtype=np.uint8)
            #image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.imread(data[path_col])
            image = cv2.cvtColor(image,cv2.COLOR_BGRA2RGB)

        row = index // cols
        col = index % cols
        show_image(ax[row,col],title,image)

    plt.show()   
sample = n_of_each(df)
sample['label'] = sample[['shape','train_or_test']].apply(lambda x: '{},{}'.format(x[0],x[1]), axis=1)
show_row_col(sample,4,path_col='path',label_col='label',mode='file')
import skimage.io
import skimage.segmentation
%matplotlib inline
plt.rcParams['figure.figsize']=10,10
# Load a single image and its associated masks
id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
file = "../input/stage1_train/{}/images/{}.png".format(id,id)
masks = "../input/stage1_train/{}/masks/*.png".format(id)
image = skimage.io.imread(file)
masks = skimage.io.imread_collection(masks).concatenate()
height, width, _ = image.shape
num_masks = masks.shape[0]

# Make a ground truth label image (pixel value is index of object label)
labels = np.zeros((height, width), np.uint16)
for index in range(0, num_masks):
    labels[masks[index] > 0] = index + 1
    
plt.imshow(image)
plt.imshow(labels,alpha=0.5)
plt.grid('False')
MASK_PATH_EXPR = os.path.join(TRAIN_PATH)+'{}/masks/{}.png'
IMAGE_PATH_EXPR = os.path.join(TRAIN_PATH)+'{}/images/{}.png'
def show_mask_and_image(image_id,mask_id,cx,cy,radius):
    mask = cv2.imread(MASK_PATH_EXPR.format(image_id,mask_id),cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(IMAGE_PATH_EXPR.format(image_id,image_id))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    fig,ax = plt.subplots(1,2,figsize=(20,20))
    ax[0].imshow(image[cy-radius:cy+radius,cx-radius:cx+radius,:])
    ax[0].set_title('image')
    ax[1].imshow(mask[cy-radius:cy+radius,cx-radius:cx+radius])
    ax[1].set_title('mask')
    ax[0].grid('False')
    ax[1].grid('False')
    plt.show()
    
image_id = '5d21acedb3015c1208b31778561f8b1079cca7487399300390c3947f691e3974'
mask_id='5e6e650a28e22f651817b2edeacbf93a960adf633f1dbef69ecea585ef35d544'
cx = 385
cy = 490
radius = 55
show_mask_and_image(image_id,mask_id,cx,cy,radius)