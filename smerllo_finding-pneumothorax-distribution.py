#------------------ util function ------------------

# credit to : https://www.kaggle.com/freeman89/eda-can-you-see-the-pneumothorax



def bounding_box(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]



    return rmin, rmax, cmin, cmax



# ---------------------------

# from mask_functions import rle2mask

# ---------------------------

def rle2mask(rle, width, height):

    mask= np.zeros(width* height)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        mask[current_position:current_position+lengths[index]] = 255

        current_position += lengths[index]



    return mask.reshape(width, height)





def plot_pixel_array(dataset, figsize=(10,10)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()

    

def plot_with_mask_and_bbox(dataset, mask_encoded, figsize=(20,10)):

    mask_decoded = rle2mask(mask_encoded, 1024, 1024).T

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))

    rmin, rmax, cmin, cmax = bounding_box(mask_decoded)

    patch = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')

    ax[0].imshow(dataset.pixel_array, cmap=plt.cm.bone)

    ax[0].imshow(mask_decoded, alpha=0.3, cmap="Reds")

    ax[0].add_patch(patch)

    ax[0].set_title('With Mask')



    patch = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')

    ax[1].imshow(dataset.pixel_array, cmap=plt.cm.bone)

    ax[1].add_patch(patch)

    ax[1].set_title('Without Mask')

    plt.show()



def show_image(img_full_path): 

    image_name= img_full_path.split('/')[-1][:-4]

    dataset = pydicom.dcmread(img_full_path)

#     show_dcm_info(dataset, image_name)

    

    mask_encoded = train_rle.set_index('ImageId').loc[image_name].values[0]

    if '-' in mask_encoded:    

        plot_pixel_array(dataset)

    else:

        plot_with_mask_and_bbox(dataset, mask_encoded)

        



def absoluteFilePaths(directory):

    ls =[]

    for dirpath,_,filenames in os.walk(directory):

        for f in filenames:

            ls.append(os.path.abspath(os.path.join(dirpath, f)))

    return ls
import os

import glob



import pydicom



from matplotlib import cm

from matplotlib import pyplot as plt

from matplotlib import patches as patches



import pandas as pd

import numpy as np

pd.options.mode.chained_assignment = None  

import sys  



ROOT_PATH=  '../input/siim-train-test/siim/'

IMAGE_PATH = ROOT_PATH +'dicom-images-train/'

IMAGE_MEDIA_TYPE = '.dcm'

IMAGE_SIZE = 1024

list_of_all_dicom_files= absoluteFilePaths(IMAGE_PATH)
train_rle = pd.read_csv(ROOT_PATH + 'train-rle.csv')

train_rle.columns

train_rle.columns= ['ImageId', 'EncodedPixels']

train_rle['Target']= [int('-' not in set(x)) for x in train_rle.EncodedPixels.values.tolist()]

img_names= train_rle[train_rle['Target']==1].ImageId.values.tolist()
x=[]

y=[]

width=[]

height=[]



train_rle['x'] = 0

train_rle['y'] = 0

train_rle['width'] = 0

train_rle['height'] = 0



from tqdm import tqdm



for img_name in tqdm(img_names):

    mask= train_rle[train_rle['ImageId']==img_name]['EncodedPixels'].values[0]

    mask = rle2mask(mask, 1024, 1024).T

    rmin, rmax, cmin, cmax = bounding_box(mask)

    train_rle['x'][train_rle['ImageId']==img_name] = 2*400-cmin 

    train_rle['y'][train_rle['ImageId']==img_name] = 2*400-rmin

    train_rle['width'][train_rle['ImageId']==img_name]= cmax-cmin

    train_rle['height'][train_rle['ImageId']==img_name]= rmax-rmin



mask_dist = train_rle[train_rle.Target==1][['x','y', 'width', 'height']]

import numpy as np

import matplotlib.pyplot as plt

plt.scatter(mask_dist.x.values, mask_dist.y.values)

plt.show()
from matplotlib.patches import Rectangle

mask_dist = train_rle[train_rle.Target==1][['x','y', 'width', 'height']]

fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))

ax1.set_xlim(0, 1024)

ax1.set_ylim(0, 1024)

for _, c_row in mask_dist.sample(1000).iterrows():

    ax1.add_patch(Rectangle(xy=(c_row['x'], c_row['y']),

                 width=c_row['width'],

                 height=c_row['height'],

                           alpha=5e-3))
X_STEPS, Y_STEPS = 1024, 1024

xx, yy = np.meshgrid(np.linspace(0, 1024, X_STEPS),

           np.linspace(0, 1024, Y_STEPS), 

           indexing='xy')

prob_image = np.zeros_like(xx)

for _, c_row in mask_dist.sample(3286).iterrows():

    c_mask = (xx>=2*400-c_row['x']) & (xx<=(2*400-c_row['x']+c_row['width']))

    c_mask &= (yy>=2*400-c_row['y']) & (yy<=2*400-c_row['y']+c_row['height'])

    prob_image += c_mask

fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))

ax1.imshow(prob_image, cmap='hot')
show_image(list_of_all_dicom_files[100])
show_image(list_of_all_dicom_files[123])
show_image(list_of_all_dicom_files[236])