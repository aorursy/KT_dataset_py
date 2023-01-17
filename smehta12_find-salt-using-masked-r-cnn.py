# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/train"))

root_dir = "../input"
train_img_dir= "train/images"
train_masks_dir = "train/masks"
tests_dir = "../input/test/images" #"../input/test/images"
#masks_csv = "..input/test/train.csv"
# Any results you write to the current directory are saved as output.

ORIG_IMG_HEIGHT=101
ORIG_IMG_WIDTH=101
# Install the matterport's masked R-CNN on Kaggle kernel
import subprocess
subprocess.call(["pip" ,"install", "git+git://github.com/matterport/Mask_RCNN.git"])
# import required package from Masked R-CNN

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
ids= ['1f1cc6b3a4','5b7c160d0d','6c40978ddf','7dfdf6eeb8','7e5a6e5013']
plt.figure(figsize=(20,10))
for j, img_name in enumerate(ids):
    q = j+1
    img = plt.imread(os.path.join(root_dir, train_img_dir, img_name + '.png'))
    img_mask = plt.imread(os.path.join(root_dir, train_masks_dir, img_name + '.png'))
    
    plt.subplot(1,2*(1+len(ids)),q*2-1)
    plt.imshow(img)
    plt.subplot(1,2*(1+len(ids)),q*2)
    plt.imshow(img_mask)
plt.show()
# Load csv
train_csv=os.path.join(root_dir, "train.csv")
train_img_info = pd.read_csv(train_csv)
train_img_info.head()
# The following parameters have been selected to reduce running time for demonstration purposes 
# These may not be optimal.

class SaltConfig(Config):
     # Give the configuration a recognizable name  
    NAME = 'find_salt'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8 
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # no mask + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 16
    MAX_GT_INSTANCES = 1
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.1
    TOP_DOWN_PYRAMID_SIZE = 128

    STEPS_PER_EPOCH = 100
    
config = SaltConfig()
config.display()
# identify whether the image has salt present or not based on the RLE masks. 
# If RLE mask present in the train df then it has salt else it doesn't has it.

train_img_info["has_salt"] = ~train_img_info["rle_mask"].isnull()
train_img_info.head()
class SaltDataSet(utils.Dataset):
    
    def __init__(self, image_files, root_dir, raw_img_dir, mask_img_dir, has_mask, orig_height, orig_width):
        super(SaltDataSet, self).__init__(self)
        
        self.add_class('salt_shape',1, 'salt')
        
        for i, image_id in enumerate(image_files):
            fp = os.path.join(root_dir, raw_img_dir, image_id+".png")
            mask_fp=os.path.join(root_dir, mask_img_dir, image_id+".png")
            self.add_image('salt_shape', image_id=i, path=fp,
                           orig_height=orig_height, orig_width=orig_width, mask_fp=mask_fp, has_mask=has_mask[i])
    
    def get_img_info(self, img_id):
        return self.image_info[img_id]
    
    def get_random_img_id(self):
        return random.choice(list(range(len(self.image_info))))

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        has_mask=info["has_mask"]
        fp = info['mask_fp']
        
        mask_img = plt.imread(fp)
        mask=np.reshape(mask_img, mask_img.shape + (1,))
        
        if not has_mask:
            class_ids=np.zeros((1,), dtype=np.int32)
        else:
            class_ids=np.ones((1,), dtype=np.int32)
        
        return mask, class_ids
image_ids=train_img_info["id"].tolist()
######################################################################
# Modify this line to use more or fewer images for training/validation. 
# To use all images, do: image_fps_list = list(image_fps)
image_id_list = list(image_ids[:1000]) 
#####################################################################

# split dataset into training vs. validation dataset 
# split ratio is set to 0.9 vs. 0.1 (train vs. validation, respectively)
validation_split = 0.1

sorted(image_id_list)
random.seed(42)
random.shuffle(image_id_list)
split_index = int((1 - validation_split) * len(image_id_list))

image_id_train = image_id_list[:split_index]
image_id_val = image_id_list[split_index:]

print(len(image_id_train), len(image_id_val))
print(image_id_train[0])
df_image_id_train = train_img_info[train_img_info["id"].isin(image_id_train)]
df_image_id_train.reset_index(drop=True)
print(df_image_id_train.shape)


df_image_id_val = train_img_info[train_img_info["id"].isin(image_id_val)]
df_image_id_val.reset_index(drop=True)
print(df_image_id_val.shape)
# Provide info about train and validation images to the SaltDataSet class

dataset_train = SaltDataSet(df_image_id_train["id"].tolist(), root_dir, train_img_dir, train_masks_dir,
                            df_image_id_train["has_salt"].tolist(), ORIG_IMG_HEIGHT, ORIG_IMG_WIDTH)
dataset_train.prepare()

dataset_val = SaltDataSet(df_image_id_val["id"].tolist(), root_dir, train_img_dir, train_masks_dir,
                            df_image_id_val["has_salt"].tolist(), ORIG_IMG_HEIGHT, ORIG_IMG_WIDTH)
dataset_val.prepare()
# show info about random image

image_id = dataset_train.get_random_img_id()
dataset_train.get_img_info(image_id)
# Load and display random samples and their bounding boxes
# Suggestion: Run this a few times to see different examples. 

image_id = dataset_train.get_random_img_id()
image_fp = dataset_train.image_reference(image_id)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)

print(dataset_train.get_img_info(image_id))

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]
plt.imshow(masked, cmap='gray')
plt.axis('off')
plt.show()
model_dir= os.path.join("model")
print(model_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
NUM_EPOCHS = [5]#, 10]

import warnings 
import time


for epoch in NUM_EPOCHS:
    current_model_dir= os.path.join(model_dir, "model_{}".format(epoch))
    print(current_model_dir)
    if not os.path.exists(current_model_dir):
        os.mkdir(current_model_dir)
    
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=current_model_dir)

    # Train Mask-RCNN Model 
    start_time = time.time()
    warnings.filterwarnings("ignore")
    
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=epoch, 
                layers='all'
               )
    end_time = time.time()

    print("completion time:{}".format((start_time-end_time)/60))
MODEL_DIR=os.path.join(model_dir, "model_{}".format(epoch[0]))

class InferenceConfig(SaltConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = model.find_last()
print(model_path)

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
image_id = random.choice(dataset_val.image_ids)
print(dataset_val.get_img_info(image_id))
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))
results = model.detect([original_image], verbose=1)


r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], figsize=(8, 8))
from itertools import groupby

def rle_encode(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    print(rle['counts'])
    return rle

print(rle_encode(r["masks"]))

plt.imshow(r["masks"].squeeze())
plt.show()
test_images=os.listdir(tests_dir)
test_images = list(map(lambda x:os.path.join(tests_dir, x), test_images))
test_images[:5]
def predict(image_paths, min_conf=0.95):
    
    submission_dict=[]
       
    # assume square image
    resize_factor = ORIG_IMG_HEIGHT / config.IMAGE_SHAPE[0]
    
    prev_mask=None
    #resize_factor = ORIG_SIZE 
    for image_id in tqdm(image_paths): 
        image = plt.imread(image_id)
        
        print(image_id)
        
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1) 
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
            
        print(image.shape)
        salt_img_id = os.path.basename(image_id).split(".")[0]
        
        results = model.detect([image])
        r = results[0]
        
        print(r["masks"].shape)
        
#         if prev_mask:
#             print("array equal:{}".format(np.array_equal(r["masks"], prev_mask)))
        prev_mask = r["masks"]
        
        plt.imshow(np.squeeze(r["masks"]))
        
        assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
        
        rle_mask=rle_encode(r['masks'])["counts"]
        
        rle_mask=" ".join(map(str, rle_mask))
                
        submission_dict.append({"id":salt_img_id, "rle_mask":rle_mask})

                               
    submission_df=pd.DataFrame(submission_dict)
    
    print(submission_df.head())
    
    return submission_df
submission_df = predict(test_images[:5])
#submission_df.to_csv(os.path.join(root_dir, "submission.csv"))