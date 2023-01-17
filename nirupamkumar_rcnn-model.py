# Importing Modules

import pandas as pd
import numpy as np
import json
import os

from PIL import Image,ImageFont
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import seaborn as sns

import os
import sys
import random
import math
import numpy as np
import skimage.io
from skimage.color import rgb2gray
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

import warnings
warnings.filterwarnings("ignore")
root_path = '/kaggle/input/imaterialist-fashion-2019-FGVC6/'
#reading train csv file
train_df = pd.read_csv(os.path.join(root_path, 'train.csv'))
train_df.shape
train_df.head(5)
train_df.info()
num_test_images = len(os.listdir(os.path.join(root_path,'test')))
num_train_images = len(os.listdir(os.path.join(root_path,'train')))
print("Number of images in test set: {}".format(num_test_images))
print("Number of images in train set: {}".format(num_train_images))
avg_class_per_image = np.round(train_df.shape[0]/num_train_images, 2)
print("Average number of classes per image: {}".format(avg_class_per_image))
assert len(train_df["ImageId"].value_counts()) == num_train_images
print("Every image has at least 1 class")
#reading categories
with open(os.path.join(root_path, 'label_descriptions.json')) as f:
    labels_data=json.load(f)
labels_data
#separating the categories and attributes
categories = pd.DataFrame(labels_data['categories'])
attributes = pd.DataFrame(labels_data['attributes'])
print("There are descriptions for", categories.shape[0],"categories and", attributes.shape[0], "attributes")
categories.head()
attributes.head()
categories['supercategory'].unique()
attributes['supercategory'].unique()
#separating categories and attributes in train data
train_df['hasAttributes'] = train_df.ClassId.apply(lambda x: x.find("_") > 0)
train_df['CategoryId'] = train_df.ClassId.apply(lambda x: x.split("_")[0]).astype(int)
train_df = train_df.merge(categories, left_on="CategoryId", right_on="id")
train_df.head()
fine_grained_obj_perc = np.round(train_df["hasAttributes"].mean()*100, 1)
print("{}% of the objects are fine-grained.".format(fine_grained_obj_perc))
fine_grained_img_perc = np.round((train_df.groupby("ImageId")["hasAttributes"].sum() > 0).mean()*100, 1)
print("{}% of the images have at least one fine-grained object.".format(fine_grained_img_perc))
class_df = train_df.groupby("CategoryId").agg({"ImageId": "count"}).reset_index()
class_df = class_df.rename(columns={"ImageId": "img_count"})
print("Number of classes: {}".format(class_df.shape[0]))
print("{} of the classes are fine-grained.".format(train_df[train_df["hasAttributes"] == True].CategoryId.nunique()))
class_df.head()
#eda
def plot_function_for_supercategories(subset,title):
    supercategory_names = np.unique(subset.supercategory)
    plt.figure(figsize=(10, 6))
    g = sns.countplot(x = 'supercategory', data=subset, order=supercategory_names)
    ax = g.axes
    tl = [x.get_text() for x in ax.get_xticklabels()]    
    ax.set_xticklabels(tl, rotation=45)
    for p, label in zip(ax.patches, supercategory_names):
        c = subset[(subset['supercategory'] == label)].shape[0]
        ax.annotate(str(c), (p.get_x()+0.3, p.get_height() + 50))
    plt.title(title)
    plt.show()
plot_function_for_supercategories(train_df[train_df.hasAttributes],'Supercategories with any attributes')
plot_function_for_supercategories(train_df[~train_df.hasAttributes],'Supercategories with no attributes')
super_cat = list(train_df['supercategory'].unique())
fig, axes = plt.subplots(6, 2, figsize=(25, 20))
z=0
for i in range(0, 6):
    for j in range(0, 2):
        sns.countplot(y="name", data=train_df[train_df.supercategory.isin([super_cat[z]])],ax = axes[i, j]).set(title = (super_cat[z]))
        fig.tight_layout()
        z=z+1
# reading sample images from training data
for i in range(6):
    id_image=train_df['ImageId'].iloc[np.random.randint(0,train_df.shape[0])]
    print('Image ID:',id_image)
    image = plt.imread(os.path.join(root_path,'train/',id_image))
    plt.imshow(image)
    plt.show()
image = plt.imread(os.path.join(root_path,'train/','b98f08f330c23af5db1c62c2412592b4.jpg'))
gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
an_array = np.where(gray_r > gray_r.mean(), 0, 3)
gray = an_array.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='binary_r')
# execution_path = '../input/imageai/resnet50_coco_best_v2.0.1.h5'
# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
# detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(root_path,'train/','b98f08f330c23af5db1c62c2412592b4.jpg'), output_image_path=os.path.join(root_path,'b98f08f330c23af5db1c62c2412592b4_detection.jpg'))
# pic = plt.imread(os.path.join(root_path,'b98f08f330c23af5db1c62c2412592b4_detection.jpg'))
# plt.imshow(pic)
# !git clone https://www.github.com/matterport/Mask_RCNN.git
# os.chdir('Mask_RCNN')
# !rm -rf .git # to prevent an error when the kernel is committed
# !rm -rf images assets # to prevent displaying images at the bottom of a kernel

# # Root directory of the project
# ROOT_DIR = os.path.abspath("../")

# import warnings
# warnings.filterwarnings("ignore")

# # Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
# from mrcnn import utils
# import mrcnn.model as modellib
# from mrcnn import visualize
# # Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco

# # Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# # Local path to trained weights file
# COCO_MODEL_PATH = os.path.join('', "mask_rcnn_coco.h5")

# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

# # Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "trump.jpg")



!pip install ProgressBar
# import required packages
from pathlib import Path
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from progressbar import ProgressBar

# create a folder for the mask images
if  not os.path.isdir('../labels'):
    os.makedirs('../labels')
train_df.columns
# path = Path("../input/imaterialist-fashion-2019-FGVC6")
# path_img = path+'/train'
# path_lbl = root_path+Path("../labels")
# only the 27 apparel items, plus 1 for background
# model image size 224x224
category_num = 27 + 1
size = 224
# get and show categories
# with open(os.path.join(root_path,"label_descriptions.json")) as f:
#     label_descriptions = json.load(f)
# label_names = [x['name'] for x in label_descriptions['categories']]
# print(label_names)
# train dataframe
df = train_df[['ImageId', 'EncodedPixels', 'Height', 'Width','ClassId']]
# training image path and images
fnames = get_image_files(os.path.join(root_path,'train'))
print(fnames[0])
# need a function to turn the run encoded pixels from train.csv into an image mask
# there are multiple rows per image for different apparel items, this groups them into one mask
def make_mask_img(segment_df):
    seg_width = segment_df.at[0, "Width"]
    seg_height = segment_df.at[0, "Height"]
    seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)
    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
        pixel_list = list(map(int, encoded_pixels.split(" ")))
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] - 1
            index_len = pixel_list[i+1] - 1
            if int(class_id.split("_")[0]) < category_num - 1:
                seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])
    seg_img = seg_img.reshape((seg_height, seg_width), order='F')
    return seg_img
# we can look at an image to see how the processing works
# the original image
img_file = fnames[500]
img = open_image(img_file)
img.show(figsize=(5,5))
# convert rows for this image into a numpy array mask
img_name = os.path.basename(img_file)
img_df = df[df.ImageId == img_name].reset_index()
#img_df = img_df.iloc[0:1]
#img_df = img_df[img_df.ClassId.astype(int) < category_num - 1].reset_index()
img_mask = make_mask_img(img_df)
plt.imshow(img_mask)
# convert the numpy array into a three channel png that can be used in the standard SegmentationItemList
# then write into the labels folder as png and show the image
# all pixels have the category numbers, so it looks like a dark greyscale image
img_mask_3_chn = np.dstack((img_mask, img_mask, img_mask))
cv2.imwrite('../labels/' + os.path.splitext(img_name)[0] + '_P.png', img_mask_3_chn)
png = open_image('../labels/' + os.path.splitext(img_name)[0] + '_P.png')
png.show(figsize=(5,5))
# use fastai's open_mask for an easier-to-view image (and check it works...)
mask = open_mask('../labels/' + os.path.splitext(img_name)[0] + '_P.png')
mask.show(figsize=(5,5), alpha=1)
print(mask.data)
# run the same procedure for a sample of first 5000 images in dataset
images = df.ImageId.unique()[:5000]
pbar = ProgressBar()
for img in pbar(images):
    img_df = df[df.ImageId == img].reset_index()
    img_mask = make_mask_img(img_df)
    img_mask_3_chn = np.dstack((img_mask, img_mask, img_mask))
    cv2.imwrite('../labels/' + os.path.splitext(img)[0] + '_P.png', img_mask_3_chn)
# before creating the databunch we need a function to find the mask images
# also set the batch size, categories and wd
get_y_fn = lambda x: Path("../labels")/f'{Path(x).stem}_P.png'
bs = 32
#classes = label_names
codes = list(range(category_num))
wd = 1e-2
# create the databunch
images_df = pd.DataFrame(images)
src = (SegmentationItemList.from_df(images_df, os.path.join(root_path,'train'))
       .split_by_rand_pct()
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
       .databunch(bs=bs)
       .normalize(imagenet_stats))
# look at a batch
data.show_batch(3, figsize=(10,10))
# I create an accuracy metric which excludes the background pixels
def acc_fashion(input, target):
    target = target.squeeze(1)
    mask = target != category_num - 1
    return (input.argmax(dim=1)==target).float().mean()
# learner, include where to save pre-trained weights (default is in non-write directory)
learn = unet_learner(data, models.resnet34, metrics=acc_fashion, wd=wd, model_dir="/kaggle/working/models")
# run learning rate finder
lr_find(learn)
learn.recorder.plot()
# set learning rate based on roughly the steepest part of the curve
lr=1e-3
# train for 10 cycles frozen
learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
# take a look at some results
learn.show_results()
# unfreeze earlier weights
learn.unfreeze()
# decrease the learning rate
lrs = slice(lr/400,lr/4)
# train for 10 more cycles unfrozen
learn.fit_one_cycle(10, lrs, pct_start=0.8)
# more results
learn.show_results()
test_path = '../input/imaterialist-fashion-2019-FGVC6/test/0046f98599f05fd7233973e430d6d04d.jpg'
img = open_image(test_path)
x = learn.predict(img)
img.show()
fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
print(learn.data.classes)
categories
test_path = '../input/imaterialist-fashion-2019-FGVC6/test/0146a53e12d690914995248fb6872121.jpg'
img = open_image(test_path)
x = learn.predict(img)
img.show()
fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
plt.imshow(x[2][27])
test_path = '../input/personal-testing/1.jpeg'
img = open_image(test_path)
x = learn.predict(img)
img.show()
fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
test_path = '../input/personal-group-1/_G2A0656.JPG'
img = open_image(test_path)
x = learn.predict(img)
img.show()
fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
test_path = '../input/personal-2/MicrosoftTeams-image (6).png'
img = open_image(test_path)
x = learn.predict(img)
img.show()

fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
test_path = '../input/personal-2/MicrosoftTeams-image (7).png'
img = open_image(test_path)
x = learn.predict(img)
img.show()

fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
plt.imshow(x[2][27])
gnames = get_image_files('../input/personal-girl-power')

gnames
test_path = gnames[0]
img = open_image(test_path)
x = learn.predict(img)
img.show()

fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
test_path = gnames[1]
img = open_image(test_path)
x = learn.predict(img)
img.show()

fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
test_path = gnames[2]
img = open_image(test_path)
x = learn.predict(img)
img.show()

fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
test_path = gnames[3]
img = open_image(test_path)
x = learn.predict(img)
img.show()

fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
test_path = gnames[4]
img = open_image(test_path)
x = learn.predict(img)
img.show()

fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
test_path = gnames[5]
img = open_image(test_path)
x = learn.predict(img)
img.show()

fig, axes = plt.subplots(9, 3, figsize=(25, 20))
z=0
for i in range(9):
    for j in range(3):
        axes[i,j].imshow(x[2][z])
#         plt.imshow(x[2][z])
        z=z+1
