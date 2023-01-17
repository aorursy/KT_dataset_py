# # Disable GPU - too small to process
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import datetime
from imgaug import augmenters as iaa
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import random
import sys
import time
import cv2

from keras.preprocessing.image import array_to_img, img_to_array, load_img

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import skimage
from sklearn.model_selection import train_test_split

import tqdm
from tqdm._tqdm_notebook import tqdm_notebook    # Progress Monitor

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
print('ROOT',ROOT_DIR)

# Data Dir where the source images live
DATA_DIR =  os.path.abspath(os.path.join(ROOT_DIR, "../../../Kaggle-Input/Airbus Ship Detection Challenge"))
print('DATA',DATA_DIR)

# Save submission files here
RESULTS_DIR = os.path.join(DATA_DIR, "results/")
print('RESULTS', RESULTS_DIR)

%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print('MODEL',MODEL_DIR)

# Import Mask RCNN
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
class ShipsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ships"
    
    DETECTION_MIN_CONFIDENCE = 0.95 

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
#     IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shape=ship

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = ShipsConfig()
config.display()
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
class ShipDataset(utils.Dataset):

    """Load a subset of the ship dataset.
    dataset_dir: Root directory of the dataset.
    subset: Subset to load: train or val
    sample:(optional)number to load
    """
    def load_ships(self, dataset_dir, subset, imageIdList, sample=None):
        exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images
        
        # Add classes. We have only one class to add.
        self.add_class("ship", 1, "ship")

        # Train or validation dataset?
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Loop
        load_count=0
        if( sample is None ):
            sample = len(imageIdList)
        for n, id_ in tqdm_notebook(enumerate(imageIdList[:sample]), total=sample):
            if( not(id_ in exclude_list)):
                self.add_image("ship", image_id=id_, path=os.path.join(dataset_dir, id_))
                load_count = load_count + 1
            if( load_count > sample):
                break
        
        # Journal 
        print('load_ships: subset',subset,' Sample=[',sample,'] Total=',load_count)
        print(dataset_dir)
        print("---")
    
    """Load an image from the ship dataset.
    image_id: filename identifying the image
    """
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = img_to_array(skimage.io.imread(self.image_info[image_id]['path']))

        # Set the height&width attributes for mask
        image_info = self.image_info[image_id]
        image_info['height'] = image.shape[0]
        image_info['width'] = image.shape[1]
#         print('shape', image.shape, 'height', image_info['height'], 'width', image_info['width'])
        
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    
    '''
    rle_decode: run-length as string formated (start length)
    mask_rle: the encoded pixel string
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    def rle_decode(self, mask_rle, shape):
        # Set class defaults
        bg_class_id = self.class_names.index("BG")
        ship_class_id = self.class_names.index("ship")

        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

        # Not all ships have masks
        #print('\n**** mask',mask_rle)
        if( pd.isnull(mask_rle) or mask_rle == 0):
            return img.reshape(shape), bg_class_id
        if( pd.isnull(mask_rle) or len(mask_rle)==0):
            return img.reshape(shape), bg_class_id

        # Split the RLE encoding into pairs
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

        # Different length
        delta=len(starts)-len(lengths)
        for x in range(0, delta): 
            lengths = np.append(lengths, [1])
            print('delta', delta, 'start', starts.shape, 'lengths', lengths.shape)

        # Bump & loop
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1

        """ CRITICAL Transpose the image mask!! """
        img = img.reshape(shape).T
#         img = Image.fromarray(img).convert('RGB')
#         print('img sum',np.sum([img]))
           
        return img, ship_class_id

    """Load a mask from the ship dataset.
            image_id: filename identifying the image
    RETURNS    masks: A bool array of shape [height, width, instance count] with one mask per instance.
           class_ids: a 1D array of class IDs of the instance masks.    
    """
    def load_mask(self, image_id):
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ship":
            return super(self.__class__, self).load_mask(image_id)

        # There may be many masks, or one, or NONE
        mask_rle = masks_df['EP'][masks_df[masks_df['ImageId'] == image_info['id']].index]
        mask_count = len(mask_rle)
        if( mask_count > 1 ):
            masks_rle = np.zeros([image_info['height'], image_info['width'], mask_count], dtype=np.uint8)
        else:
            masks_rle = np.zeros([image_info['height'], image_info['width'], 1], dtype=np.uint8)
        
        class_ids = np.arange(1, dtype=int)

        for i, mask_str in enumerate(mask_rle):
            # First get the rle encoding string
            mask_rle, class_id = self.rle_decode( mask_str, (768,768) )
        
            # Now resize the 768 image
            info = self.image_info[image_id]
            mask_rle = np.resize(mask_rle, (image_info['height'], image_info['width'], 1))
#             print('mask shape',mask_rle.shape,'count',mask_count,'for class', class_id, self.class_names[class_id])
        
            # Journal the mask into the return array
            for h in range(0, mask_rle.shape[0]):
                for w in range(0, mask_rle.shape[1]):
                    if( mask_rle[h][w] == 1 ):
#                         masks_rle[h][w][i:i+1] = 1
                        masks_rle[h][w][i] = 1
        
            # And resize the class array
            if( mask_count > 0 ):
                class_ids = np.arange(mask_count, dtype=int)
            # stuff the arrays with the returned classid
            class_ids = np.full_like(class_ids, class_id)

        # Return mask, and a ship class ID
        return masks_rle.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

masks_df = pd.read_csv( os.path.join(ROOT_DIR, 'train_ship_segmentations.csv') )
masks_df['ships'] = masks_df['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
masks_df['has_ship'] = masks_df['ships']
# masks_df['EP'] = masks_df['EncodedPixels'] + ' '
masks_df['EP'] = masks_df['EncodedPixels'].map(lambda c_row: ' '+c_row if isinstance(c_row, str) else '')

print(masks_df.shape[0], 'masks_df found')
print(masks_df['ImageId'].value_counts().shape[0], 'unique images')
masks_df.head(20)
unique_img_ids = masks_df.groupby(by='ImageId', group_keys=True).agg( {'ships':'sum', 'has_ship':'max'} ).reset_index()
print(unique_img_ids.shape[0], 'unique_img_ids found')
unique_img_ids.head(20)
unique_img_ids[['has_ship','ships']].hist()
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images

for e1 in exclude_list:
    unique_img_ids.drop( unique_img_ids[unique_img_ids['ImageId'] == e1].index, inplace=True)

print(unique_img_ids.shape[0], 'training and validation images')
# unique_img_ids.drop( unique_img_ids[unique_img_ids['ships'] == 0].index, inplace=True)

# print(unique_img_ids.shape[0], 'training and validation images')
train_ids, valid_ids = train_test_split(unique_img_ids, test_size = 0.3#)
                                        ,stratify = unique_img_ids['ships'])
train_df = pd.merge(unique_img_ids, train_ids)
valid_df = pd.merge(unique_img_ids, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')
# Training dataset
dataset_train = ShipDataset()
dataset_train.load_ships(dataset_dir=DATA_DIR, subset="train", imageIdList=train_df['ImageId'])
dataset_train.prepare()

# Validation dataset
dataset_val = ShipDataset()
dataset_val.load_ships(dataset_dir=DATA_DIR, subset="train", imageIdList=valid_df['ImageId'])
dataset_val.prepare()
# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
print('image_ids',image_ids)
        
for image_id in image_ids:
#     print(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, 1)
# Load and display random samples
image_ids = np.random.choice(dataset_val.image_ids, 4)
print('image_ids',image_ids)
        
for image_id in image_ids:
#     print(image_id)
    image = dataset_val.load_image(image_id)
    mask, class_ids = dataset_val.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, 1)
# Load random image and mask.
image_id = np.random.choice(dataset_val.image_ids, 1)[0]
image = dataset_val.load_image(image_id)
mask, class_ids = dataset_val.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset_train.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
# init_with = "imagenet"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
# Image augmentation
# http://imgaug.readthedocs.io/en/latest/source/augmenters.html
augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
               iaa.Affine(rotate=180),
               iaa.Affine(rotate=270)]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])
# If starting from imagenet, train heads only for a bit
# since they have random weights
print("Train network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            augmentation=augmentation,
            layers='heads')

# model.train(dataset_train, dataset_val, 
#             learning_rate=config.LEARNING_RATE, 
#             epochs=1, 
#             layers='heads')
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
print("Train all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            augmentation=augmentation,
            layers='all')

# model.train(dataset_train, dataset_val, 
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=2, 
#             layers="all")
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
model.keras_model.save_weights(model_path)
class InferenceConfig(ShipsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))
results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))
# Test dataset
submission_df = pd.read_csv('sample_submission.csv')
submission_df.head(20)
dataset_test = ShipDataset()
dataset_test.load_ships(dataset_dir=DATA_DIR, subset="test", imageIdList=submission_df['ImageId'])
dataset_test.prepare()
def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))

def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

# Create directory
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
submit_dir = os.path.join(RESULTS_DIR, submit_dir)
os.makedirs(submit_dir)
print('Submission results in',submit_dir)

# Load over images
submission = []
for image_id in tqdm_notebook(dataset_test.image_ids):
    # Load image and run detection
    image = dataset_test.load_image(image_id)
    
    # Detect objects
    r = model.detect([image], verbose=0)[0]
    
    # Encode image to RLE. Returns a string of multiple lines
    source_id = dataset_test.image_info[image_id]["id"]
#     rle = mask_to_rle(source_id, r["masks"], r["scores"])
#     submission.append(rle)
    num_instances = len(r['rois'])

    for i in range(num_instances):
        mi = r["masks"][...,i]
        mi = np.reshape(mi, (mi.shape[0],mi.shape[1],1))
        if r['scores'][i] > config.DETECTION_MIN_CONFIDENCE:
            rle = mask_to_rle(source_id, mi, r["scores"][i])
            submission.append(rle)
#     # Save image with masks
#     visualize.display_instances(
#         image, r['rois'], r['masks'], r['class_ids'],
#         dataset_test.class_names, r['scores'],
#         show_bbox=False, show_mask=False,
#         title="Predictions "+source_id)
#     plt.savefig("{}/{}.png".format(submit_dir, dataset_test.image_info[image_id]["id"]))

# Save to csv file
submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
file_path = os.path.join(submit_dir, "submit.csv")
with open(file_path, "w") as f:
    f.write(submission)
print("Saved to ", submit_dir)
print(submission[:835])
