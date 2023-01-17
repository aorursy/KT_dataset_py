# Basics

from glob import glob # finds pathnames

import os # Miscellaneous operating system interfaces

import sys

import random

import timeit

import imp

import gc



# Processing

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.ndimage import label as scipy_label

from scipy.ndimage import generate_binary_structure



# Image manipulation

import skimage



# Import Mask RCNN

sys.path.append('../input/mask-rcnn')  # To find local version of the library

from config import Config

# Imp import to ensure loading the correct utils package

fp, pathname, description = imp.find_module('utils',['../input/mask-rcnn'])

utils = imp.load_module('utils',fp,pathname,description)

import model as modellib

import visualize

from model import log



# Plotting

import matplotlib.pyplot as plt
def get_ax(rows=1, cols=1, size=8):

    """Return a Matplotlib Axes array to be used in

    all visualizations in the notebook. Provide a

    central point to control graph sizes.

    

    Change the default size attribute to control the size

    of rendered images

    """

    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))

    return ax
# Paths

ROOT_DIR = '../input/severstal-steel-defect-detection/'

Train_Dir = ROOT_DIR + 'train_images/'

Test_Dir = ROOT_DIR + 'test_images/'



# Directory to save logs and trained model

MODEL_DIR = 'logs'



# Local path to trained weights file

# Trained_Weights = '../input/????.h5'
class SteelConfig(Config):

    """Configuration for training on the steel dataset.

    Derives from the base Config class and overrides values specific

    to the steel dataset.

    """

    # Give the configuration a recognizable name

    NAME = "steel"



    # Train on 1 GPU and 1 image per GPU.

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1



    # Number of classes (including background)

    NUM_CLASSES = 1 + 4  # background + 4 defect classes



    # Use small images (128x128) for faster training. Set the limits of the small side

    # the large side, and that determines the image shape.

    IMAGE_MIN_DIM = 256

    IMAGE_MAX_DIM = 256



    # Use smaller anchors because our objects tend to be small 

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels



    # Reduce training ROIs per image because the images have few objects.

    # Aim to allow ROI sampling to pick 33% positive ROIs.

    TRAIN_ROIS_PER_IMAGE = 32



    # Minimum probability value to accept a detected instance

    # ROIs below this threshold are skipped

    DETECTION_MIN_CONFIDENCE = 0.7

    

    # Number of epochs

    EPOCHS = 3

    

    # Steps per epoch

    STEPS_PER_EPOCH = 100



    # validation steps per epoch

    VALIDATION_STEPS = 5

    

    # Non-maximum suppression threshold for detection, default 0.3

    DETECTION_NMS_THRESHOLD = 0.0

    

    # Non-max suppression threshold to filter RPN proposals. default 0.7

    # You can increase this during training to generate more propsals.

    RPN_NMS_THRESHOLD = 0.0

    

config = SteelConfig()

config.display()
class SteelDataset(utils.Dataset):



    def load_steel(self, dataset_dir, files):

        """Load a subset of the Steel dataset.

        

        Input:

        dataset_dir: Root directory of the dataset.

        files: filenames of images to load

        

        Creates:

        image objects:

            source: source label

            image_id: id, used filename

            path: path + filename

            rle: rle mask encoded pixels, required for mask conversion

            classes: classes for the rle masks, required for mask conversion        

        """

        # Add classes.

        self.add_class("steel", 1, "defect1")

        self.add_class("steel", 2, "defect2")

        self.add_class("steel", 3, "defect3")

        self.add_class("steel", 4, "defect4")

        

        # Load annotations CSV

        annotations_train = pd.read_csv(dataset_dir + 'train.csv')



        # Remove images without Encoding

        annotations_train_Encoded = annotations_train[annotations_train['EncodedPixels'].notna()].copy()        

        

        # Split ImageId_ClassId

        ImageId_ClassId_split = annotations_train_Encoded["ImageId_ClassId"].str.split("_", n = 1, expand = True)

        annotations_train_Encoded['ImageId'] = ImageId_ClassId_split.loc[:,0]

        annotations_train_Encoded['ClassId'] = ImageId_ClassId_split.loc[:,1]     



        for file in files:

            EncodedPixels = [annotations_train_Encoded['EncodedPixels'][annotations_train_Encoded['ImageId'] == file]]

            ClassID = (annotations_train_Encoded['ClassId'][annotations_train_Encoded['ImageId'] == file])



            self.add_image(

                source = "steel",

                image_id = file,  # use filename as a unique image id

                path = Train_Dir + '/' + file,

                rle = EncodedPixels,

                classes = ClassID)



    def load_mask(self, image_id):

        """Generate instance masks for an image.

        Input:

        image_id: id of the image

        

        Returns:

        masks: A bool array of shape [height, width, instance count] with one mask per instance.

        class_ids: a 1D int array of class IDs of the instance masks.

        """

        # If not a steel dataset image, delegate to parent class.

        image_info = self.image_info[image_id]

        if image_info["source"] != "steel":

            return super(self.__class__, self).load_mask(image_id)

        

        # Convert rle to single mask

        ClassIDIndex = 0

        ClassID = np.empty(0, dtype = int)

        maskarray = np.empty((256, 1600, 0), dtype = int)

        for rlelist in image_info['rle']:

            for row in rlelist:

                mask= np.zeros(1600 * 256 ,dtype=np.uint8)

                array = np.asarray([int(x) for x in row.split()])

                starts = array[0::2]-1

                lengths = array[1::2]    

                for index, start in enumerate(starts):

                    mask[int(start):int(start+lengths[index])] = 1

                mask = mask.reshape((256,1600), order='F')

                # Label mask elements

                structure = generate_binary_structure(2,2)

                labeled_array, labels = scipy_label(mask, structure)

                # Convert labeled_array elements to bitmap mask array

                for label in range(labels):

                    labelmask = np.copy(labeled_array)    

                    labelmask[labelmask != label + 1] = 0

                    if label == 0:

                        labelmask = np.expand_dims(labelmask, axis = 2)

                        maskarray = np.concatenate((maskarray, labelmask), axis = 2)

                    else:

                        labelmask[labelmask == label + 1] = 1

                        labelmask = np.expand_dims(labelmask, axis = 2)

                        maskarray = np.concatenate((maskarray, labelmask), axis = 2)

                    # Update ClassID list

                    ClassID = np.append(ClassID,int(image_info['classes'].iloc[ClassIDIndex]))

                ClassIDIndex = ClassIDIndex + 1



        # Return mask, and array of class IDs of each instance.

        return maskarray.astype(np.bool), ClassID



    def image_reference(self, image_id):

        """Return the path of the image."""

        info = self.image_info[image_id]

        if info["source"] == "steel":

            return info["path"]

        else:

            super(self.__class__, self).image_reference(image_id)
# Load example images and masks.

files = ['ef24da2ba.jpg', 'db4867ee8.jpg', 'a28a7b7be.jpg', 'c44784905.jpg']

dataset = SteelDataset()

dataset.load_steel(ROOT_DIR, files)

dataset.prepare()



image_ids = [0,1,2,3] 

for image_id in image_ids:

    image = dataset.load_image(image_id)

    mask, class_ids = dataset.load_mask(image_id)

    # Compute Bounding box

    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats

    print("image_id ", image_id, dataset.image_reference(image_id))

    log("image", image)

    log("mask", mask)

    log("class_ids", class_ids)

    log("bbox", bbox)

    # Display image and instances

    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
# select files for test and validation dataset

# Load annotations CSV

annotations_train = pd.read_csv(ROOT_DIR + 'train.csv')



# Remove images without Encoding

annotations_train_Encoded = annotations_train[annotations_train['EncodedPixels'].notna()].copy()        



# Split ImageId_ClassId

ImageId_ClassId_split = annotations_train_Encoded["ImageId_ClassId"].str.split("_", n = 1, expand = True)

annotations_train_Encoded['ImageId'] = ImageId_ClassId_split.loc[:,0]

annotations_train_Encoded['ClassId'] = ImageId_ClassId_split.loc[:,1]



# Split dataframe

msk = np.random.rand(len(annotations_train_Encoded)) < 0.85

train_msk = annotations_train_Encoded[msk]

val_msk = annotations_train_Encoded[~msk]

train = train_msk['ImageId'].unique().copy()

val = val_msk['ImageId'].unique().copy()

print('Test images: ' + str(len(train)))

print('Val images: ' + str(len(val)))

# Cleanup

del annotations_train, ImageId_ClassId_split, annotations_train_Encoded, msk, train_msk, val_msk 

gc.collect()
# Training preperations

# Training dataset

dataset_train = SteelDataset()

dataset_train.load_steel(ROOT_DIR, train)

dataset_train.prepare()



# Validation dataset

dataset_val = SteelDataset()

dataset_val.load_steel(ROOT_DIR, val)

dataset_val.prepare()



# Build training model

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Required due to change in new Tensorflow / Keras version

model.keras_model.metrics_tensors = []

# Load weights to continue training

# model.load_weights(Trained_Weights, by_name=True)
# Train model

timestart = timeit.default_timer()

print("Training")

model.train(dataset_train, 

            dataset_val,

            learning_rate=config.LEARNING_RATE,

            epochs=config.EPOCHS,

            layers='all')

timestop = timeit.default_timer()

runtime = np.round((timestop - timestart) / 60, 2)

print ('Total run time: ' + str(runtime) + ' minutes')
# Plot loss

history = model.keras_model.history.history

epochs = range(1,config.EPOCHS + 1)

plt.figure(figsize=(10, 5))

plt.plot(epochs, history['loss'], label="train loss")

plt.plot(epochs, history['val_loss'], label="valid loss")

plt.legend()

plt.show()