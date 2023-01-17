# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import sys

#setting up Mask R-CNN

!git clone https://github.com/matterport/Mask_RCNN.git

!pip install tensorflow==1.14

!pip install swifter

!pip install keras==2.2.4

ROOT_DIR = '/kaggle/working'

sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library

from mrcnn.config import Config

from mrcnn import utils

import mrcnn.model as modellib

from mrcnn import visualize

from mrcnn.model import log

from os import listdir

from os.path import isfile, join

import sys

import random

import math

import re

import time

import glob

import json

import collections



import numpy as np

import cv2

import skimage

import pandas as pd

import tensorflow as tf

from mrcnn.config import Config

import numpy as np

from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

import matplotlib.lines as lines

from matplotlib.patches import Polygon

import matplotlib.patches as patches





from mrcnn import utils

from mrcnn import visualize

from mrcnn.visualize import display_images

import imgaug.augmenters as iaa

import imgaug

#initialize wandb

import wandb

import sys

import wandb

import sys



import swifter

import glob

import re

import os

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session







all_direcs = glob.glob("/kaggle/input/lisa-traffic-light-dataset/*")



all_files_annotation = []

for dirname, _, filenames in os.walk('/kaggle/input/lisa-traffic-light-dataset/Annotations'):

    for filename in filenames:

        if (("csv" in filename) & ("BOX" in filename)):

            all_files_annotation.append(os.path.join(dirname, filename))

# ['/kaggle/input/lisa-traffic-light-dataset/Annotations']

all_direcs_images = ['/kaggle/input/lisa-traffic-light-dataset/dayTrain',

 '/kaggle/input/lisa-traffic-light-dataset/nightSequence2',

 '/kaggle/input/lisa-traffic-light-dataset/daySequence1',

 '/kaggle/input/lisa-traffic-light-dataset/nightTrain',

 '/kaggle/input/lisa-traffic-light-dataset/daySequence2',

 '/kaggle/input/lisa-traffic-light-dataset/nightSequence1']



all_files_images = []

for direc in all_direcs_images:

    for dirname, _, filenames in os.walk(direc):

        for filename in filenames:

            if ".jpg" in filename:

                all_files_images.append(os.path.join(dirname, filename))



all_files_images



#check schema of all the CSVs

all_schema = [pd.read_csv(f).columns for f in all_files_annotation]

for i in range(len(all_schema) -1):

    if all_schema[i] == all_schema[i+1]:

        print("OK!")

    else:

        print("Not OK, with id as ", i)



df_annotation = pd.concat([pd.read_csv(f, sep=";") for f in all_files_annotation])

df_annotation.index = range(df_annotation.shape[0])

df_annotation["image_name"] = df_annotation["Filename"].apply(lambda x: x.split("/")[-1])

from mrcnn.model import MaskRCNN

from wandb.keras import WandbCallback
import keras
class CustomCallback(keras.callbacks.Callback):

    def __init__(self, direc):

        """ Save params in constructor

        """

        self.direc = direc

    def on_epoch_end(self, epoch, logs=None):

        keys = list(logs.keys())

        

        self.model.save_weights(os.path.join(self.direc, "mask_rcnn_{epoch:02d}_object_detection.h5"))

        wandb.save(os.path.join(self.direc, "mask_rcnn_{epoch:02d}_object_detection.h5"))

        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

class MaskRCNN(MaskRCNN):

    

    

    def set_log_dir(self, model_path=None):

        """Sets the model log directory and epoch counter.



        model_path: If None, or a format different from what this code uses

            then set a new log directory and start epochs from 0. Otherwise,

            extract the log directory and the epoch counter from the file

            name.

        """

        # Set date and epoch counter as if starting a new model

        self.epoch = 0

        now = datetime.datetime.now()



        # If we have a model path with date and epochs use them

        if model_path:

            # Continue from we left of. Get epoch and date from the file name

            # A sample model path might look like:

            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)

            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)

            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"

            m = re.match(regex, model_path)

            if m:

                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),

                                        int(m.group(4)), int(m.group(5)))

                # Epoch number in file is 1-based, and in Keras code it's 0-based.

                # So, adjust for that then increment by one to start from the next epoch

                self.epoch = int(m.group(6)) - 1 + 1

                print('Re-starting from epoch %d' % self.epoch)



        # Directory for training logs

        self.log_dir = wandb.run.dir



        # Path to save after each epoch. Include placeholders that get filled by Keras.

        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(

            self.config.NAME.lower()))

        self.checkpoint_path = self.checkpoint_path.replace(

            "*epoch*", "{epoch:04d}")    

    

    

    

    

    

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,

              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):

        """Train the model.

        train_dataset, val_dataset: Training and validation Dataset objects.

        learning_rate: The learning rate to train with

        epochs: Number of training epochs. Note that previous training epochs

                are considered to be done alreay, so this actually determines

                the epochs to train in total rather than in this particaular

                call.

        layers: Allows selecting wich layers to train. It can be:

            - A regular expression to match layer names to train

            - One of these predefined values:

              heads: The RPN, classifier and mask heads of the network

              all: All the layers

              3+: Train Resnet stage 3 and up

              4+: Train Resnet stage 4 and up

              5+: Train Resnet stage 5 and up

        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)

            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)

            flips images right/left 50% of the time. You can pass complex

            augmentations as well. This augmentation applies 50% of the

            time, and when it does it flips images right/left half the time

            and adds a Gaussian blur with a random sigma in range 0 to 5.



                augmentation = imgaug.augmenters.Sometimes(0.5, [

                    imgaug.augmenters.Fliplr(0.5),

                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))

                ])

        custom_callbacks: Optional. Add custom callbacks to be called

            with the keras fit_generator method. Must be list of type keras.callbacks.

        no_augmentation_sources: Optional. List of sources to exclude for

            augmentation. A source is string that identifies a dataset and is

            defined in the Dataset class.

        """

        assert self.mode == "training", "Create model in training mode."



        # Pre-defined layer regular expressions

        layer_regex = {

            # all layers but the backbone

            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",

            # From a specific Resnet stage and up

            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",

            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",

            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",

            # All layers

            "all": ".*",

        }

        if layers in layer_regex.keys():

            layers = layer_regex[layers]



        # Data generators

        train_generator = data_generator(train_dataset, self.config, shuffle=True,

                                         augmentation=augmentation,

                                         batch_size=self.config.BATCH_SIZE,

                                         no_augmentation_sources=no_augmentation_sources)

        val_generator = data_generator(val_dataset, self.config, shuffle=True,

                                       batch_size=self.config.BATCH_SIZE)



        # Create log_dir if it does not exist

        if not os.path.exists(self.log_dir):

            os.makedirs(self.log_dir)

        filepath = os.path.join(self.log_dir, "/mask_rcnn_{epoch:02d}-{val_loss:.2f}.h5")

        # Callbacks

        callbacks = [

            keras.callbacks.TensorBoard(log_dir=self.log_dir,

                                        histogram_freq=0, write_graph=True, write_images=False),

#             WandbCallback(data_type="image", labels=CLASSES_TO_TAKE, save_weights_only=True),

#             CustomCallback(direc=self.log_dir)

#             keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True)

        ]



        # Add custom callbacks to the list

        if custom_callbacks:

            callbacks += custom_callbacks



        # Train

        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))

        log("Checkpoint Path: {}".format(self.checkpoint_path))

        self.set_trainable(layers)

        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)



        # Work-around for Windows: Keras fails on Windows when using

        # multiprocessing workers. See discussion here:

        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009

        if os.name is 'nt':

            workers = 0

        else:

            workers = multiprocessing.cpu_count()



        self.keras_model.fit_generator(

            train_generator,

            initial_epoch=self.epoch,

            epochs=epochs,

            steps_per_epoch=self.config.STEPS_PER_EPOCH,

            callbacks=callbacks,

            validation_data=val_generator,

            validation_steps=self.config.VALIDATION_STEPS,

            max_queue_size=100,

            workers=workers,

            use_multiprocessing=True,

        )

        self.epoch = max(self.epoch, epochs)



    
ASSET_LIST = df_annotation["Annotation tag"].unique()

CLASSES_TO_TAKE = ASSET_LIST

TRAINING_NAME = "traffic-sign-object-detection"

le = LabelEncoder()



asset_to_classes = {}

le_fit =  le.fit(CLASSES_TO_TAKE)



for el in CLASSES_TO_TAKE:

    asset_to_classes[el] = le_fit.transform([el])[0] +1

    



classes_to_asset = {}

for k,v in asset_to_classes.items():

    classes_to_asset[v] = k
from kaggle_secrets import UserSecretsClient

secret_label = "wandb_key"

secret_value = UserSecretsClient().get_secret(secret_label)

os.environ["WANDB_API_KEY"] = secret_value
wandb.login()


def train_sweep():

    wandb.init(sync_tensorboard =True)

    # Specify the hyperparameter to be tuned along with

    # an initial value

    configs = {

        'RPN_ANCHOR_SCALES': (8, 16, 32, 64, 128)

    }

    

    # Specify the other hyperparameters to the configuration

    config = wandb.config

    config.epochs = 2

    

    # Add the config item (layers) to wandb

    if wandb.run:

        wandb.config.update({k: v for k, v in configs.items() if k not in dict(wandb.config.user_items())})

        configs = dict(wandb.config.user_items())    

        

    class ShapesConfig(Config):

        """Configuration for training on the toy shapes dataset.

        Derives from the base Config class and overrides values specific

        to the toy shapes dataset.

        """

        # Give the configuration a recognizable name

        NAME = TRAINING_NAME



        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each

        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).

        GPU_COUNT = 1

        IMAGES_PER_GPU = 4



        # Number of classes (including background)

        NUM_CLASSES = 1 + len(CLASSES_TO_TAKE)  # background + 2 types of cracks



        # Use small images for faster training. Set the limits of the small side

        # the large side, and that determines the image shape.

        IMAGE_MIN_DIM = 256

        IMAGE_MAX_DIM = 256



        # Use smaller anchors because our image and objects are small

        RPN_ANCHOR_SCALES = wandb.config.RPN_ANCHOR_SCALES  # anchor side in pixels

        RPN_NMS_THRESHOLD = wandb.config.RPN_NMS_THRESHOLD

        MEAN_PIXEL = np.array([101.23385732682024, 101.23385732682024, 101.23385732682024])

        # Reduce training ROIs per image because the images are small and have

        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.

        TRAIN_ROIS_PER_IMAGE = 29

        USE_MINI_MASK = True

        # Use a small epoch since the data is simple

        STEPS_PER_EPOCH = 7



        # use small validation steps since the epoch is small

        VALIDATION_STEPS = 10

        RPN_ANCHOR_RATIOS = [.5, 1, 2]



    network_config = ShapesConfig()

    dataset_train = AsphaltShapeCracksDataset()



    dataset_train.load_crack(subset="train")



    dataset_train.prepare()



    # validation dataset

    dataset_val = AsphaltShapeCracksDataset()



    dataset_val.load_crack(subset="val")



    dataset_val.prepare()



    model = modellib.MaskRCNN(mode="training", config=network_config, model_dir="logs")



    init_with = "coco"  # imagenet, coco, or last



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

    

    

    

#     filepath = os.path.join(wandb.run.dir, "/mask_rcnn_{epoch:02d}-{val_loss:.2f}.h5")

    

    

    custom_callbacks = [WandbCallback(data_type="image", labels=CLASSES_TO_TAKE, save_weights_only=True), 

#                         keras.callbacks.ModelCheckpoint(os.path.join(wandb.run.dir, "mask_rcnn_{epoch:02d}_object_detection.h5"))

                        CustomCallback(direc=wandb.run.dir)

#                        keras.callbacks.ModelCheckpoint(filepath)

                       ]

    model.train(dataset_train, dataset_val, 

                learning_rate=0.001, 

                epochs=config.epochs, 

                layers='heads', custom_callbacks = custom_callbacks)
# config_attrs = ['BACKBONE',

#  'BACKBONE_STRIDES',

#  'BATCH_SIZE',

#  'BBOX_STD_DEV',

#  'COMPUTE_BACKBONE_SHAPE',

#  'DETECTION_MAX_INSTANCES',

#  'DETECTION_MIN_CONFIDENCE',

#  'DETECTION_NMS_THRESHOLD',

#  'FPN_CLASSIF_FC_LAYERS_SIZE',

#  'GPU_COUNT',

#  'GRADIENT_CLIP_NORM',

#  'IMAGES_PER_GPU',

#  'IMAGE_CHANNEL_COUNT',

#  'IMAGE_MAX_DIM',

#  'IMAGE_META_SIZE',

#  'IMAGE_MIN_DIM',

#  'IMAGE_MIN_SCALE',

#  'IMAGE_RESIZE_MODE',

#  'IMAGE_SHAPE',

#  'LEARNING_MOMENTUM',

#  'LEARNING_RATE',

#  'LOSS_WEIGHTS',

#  'MASK_POOL_SIZE',

#  'MASK_SHAPE',

#  'MAX_GT_INSTANCES',

#  'MEAN_PIXEL',

#  'MINI_MASK_SHAPE',

#  'NAME',

#  'NUM_CLASSES',

#  'POOL_SIZE',

#  'POST_NMS_ROIS_INFERENCE',

#  'POST_NMS_ROIS_TRAINING',

#  'PRE_NMS_LIMIT',

#  'ROI_POSITIVE_RATIO',

#  'RPN_ANCHOR_RATIOS',

#  'RPN_ANCHOR_SCALES',

#  'RPN_ANCHOR_STRIDE',

#  'RPN_BBOX_STD_DEV',

#  'RPN_NMS_THRESHOLD',

#  'RPN_TRAIN_ANCHORS_PER_IMAGE',

#  'STEPS_PER_EPOCH',

#  'TOP_DOWN_PYRAMID_SIZE',

#  'TRAIN_BN',

#  'TRAIN_ROIS_PER_IMAGE',

#  'USE_MINI_MASK',

#  'USE_RPN_ROIS',

#  'VALIDATION_STEPS',

#  'WEIGHT_DECAY']

# config_dict = dict()

# for attr in config_attrs:

#     config_dict[attr] = getattr(network_config, attr)
# config.display()
from datetime import datetime
# now= datetime.now()



# name = "_".join([str(f) for f in [now.day, now.month, now.year, now.minute, now.second, now.microsecond]])

# # wandb.init(anonymous='allow', project="kaggle-feature-encoding", name=name, reinit=True)

# wandb.init(project=TRAINING_NAME, name=name, config = config_dict, sync_tensorboard =True)

# # callbacks=[WandbCallback(data_type="image", labels=CLASSES_TO_TAKE), TensorBoard(log_dir=wandb.run.dir)]
df_all_imgs = pd.DataFrame(all_files_images, columns = ["image_path"])

df_all_imgs["image_name"] = df_all_imgs["image_path"].apply(lambda x: x.split("/")[-1])





def find_width_height(x):

    ls = cv2.imread(str(x["image_path"])).shape

    return ls



df_all_imgs["height-width"] = df_all_imgs.apply(lambda x: (960, 1280, 3), axis=1)



df_annotation_final = pd.merge(df_annotation,df_all_imgs, on=["image_name"], how="left")
df_annotation_final.to_pickle("df_annotation.pickle")
image_with_signs = df_annotation_final["image_name"].unique().tolist()
ls_img = [f for f in all_files_images if f.split("/")[-1] in image_with_signs]



valid_names = np.random.choice(ls_img,int(.10*(len(ls_img))), replace=False).tolist()

train_names = list(set(ls_img) - set(valid_names))

len(valid_names), len(train_names)
# config.display()
def img_to_annotation(df_frame):

    

    target_mat = []

    class_matrix = []

    for r in df_frame.iterrows():

        anno_matrix  = np.zeros((256, 256))

        c = r[1]["Annotation tag"]

        xmin, ymin, xmax, ymax = r[1]["Upper left corner X"],  r[1]["Upper left corner Y"],  r[1]["Lower right corner X"],  r[1]["Lower right corner Y"]

        height, width, _ = r[1]["height-width"]

        xmin = int(256.*(xmin/float(width)))

        xmax = int(256.*(xmax/float(width)))

        ymin = int(256.*(ymin/height))

        ymax = int(256.*(ymax/height))

        anno_matrix[ymin: ymax, xmin: xmax] =1

        class_matrix.append(asset_to_classes[c])

        target_mat.append(anno_matrix)

    

    if (len(class_matrix) == 0):

        pass

    elif (len(class_matrix) ==1):

        mask = target_mat[0]

        mask = np.reshape(mask,(256,256,1))

        mask = np.array(mask)

     

    else:

        mask = np.concatenate((np.expand_dims(target_mat[0],axis=-1), np.expand_dims(target_mat[1],axis=-1)),axis =-1)

        for j in range(2,len(class_matrix)):

            mask = np.concatenate((mask,np.expand_dims(target_mat[j],-1)),axis = -1)     

    return mask,class_matrix
class AsphaltShapeCracksDataset(utils.Dataset):

    """Generates the shapes synthetic dataset. The dataset consists of simple

    shapes (triangles, squares, circles) placed randomly on a blank surface.

    The images are generated on the fly. No file access required.

    """



    def load_crack(self, subset):

        """Generate an image from the specs of the given image ID.

        Typically this function loads the image from a file, but

        in this case it generates the image on the fly from the

        specs in image_info.

        """

        for a_id, asset in enumerate(ASSET_LIST):

            self.add_class(TRAINING_NAME, asset_to_classes[asset], asset)



      

        assert subset in ["train", "val"]

        if subset == "val":

            image_ids = valid_names

        elif subset == "train":

            image_ids = train_names

        height, width = (256,256)

        

        for image_id in image_ids:

            self.add_image(

                TRAINING_NAME,

                image_id=image_id.split("/")[-1],

                path= image_id)



    def image_reference(self, image_id):

        """Return the shapes data of the image."""

        info = self.image_info[image_id]

        if info["source"] == TRAINING_NAME:

            return info["path"]

        else:

            super(self.__class__).image_reference(self, image_id)            



    def load_mask(self, image_id):

        """Generate instance masks for shapes of the given image ID.

        """

        info = self.image_info[image_id]

#         print(info["id"])

        data_frame = df_annotation_final[df_annotation_final["image_name"] == info["id"]]

        mask, class_ids = img_to_annotation(data_frame)    

        class_ids = np.array(class_ids)

        return mask.astype(np.bool), class_ids.astype(np.int32)  

    

    def load_image(self, image_id):

        """Load the specified image and return a [H,W,3] Numpy array.

        """

        # Load image

        image = cv2.imread(self.image_info[image_id]['path'])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        # If grayscale. Convert to RGB for consistency.

        if image.ndim != 3:

            image = skimage.color.gray2rgb(image)

        # If has an alpha channel, remove it for consistency

        if image.shape[-1] == 4:

            image = image[..., :3]

        

        return image


# dataset_train = AsphaltShapeCracksDataset()



# dataset_train.load_crack(subset="train")



# dataset_train.prepare()



# # validation dataset

# dataset_val = AsphaltShapeCracksDataset()



# dataset_val.load_crack(subset="val")



# dataset_val.prepare()
# plt.imshow(dataset_train.load_image(0))
# plt.imshow(dataset_train.load_mask(0)[0][:,:,1])
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed

if not os.path.exists(COCO_MODEL_PATH):

    utils.download_trained_weights(COCO_MODEL_PATH)
# model = modellib.MaskRCNN(mode="training", config=network_config, model_dir="logs")



# init_with = "coco"  # imagenet, coco, or last



# if init_with == "imagenet":

#     model.load_weights(model.get_imagenet_weights(), by_name=True)

# elif init_with == "coco":

#     # Load weights trained on MS COCO, but skip layers that

#     # are different due to the different number of classes

#     # See README for instructions to download the COCO weights

#     model.load_weights(COCO_MODEL_PATH, by_name=True,

#                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 

#                                 "mrcnn_bbox", "mrcnn_mask"])

# elif init_with == "last":

#     # Load the last model you trained and continue training

#     model.load_weights(model.find_last(), by_name=True)



# model.train(dataset_train, dataset_val, 

#             learning_rate=0.001, 

#             epochs=2, 

#             layers='heads')
sweep_config = {

   'method': 'grid',

   'parameters': {

       'RPN_ANCHOR_SCALES': {

           'values': [(8, 16, 32, 64, 128),

                     (4,8,16,32,64)]

       },

       'RPN_NMS_THRESHOLD':{

           'values':[0.7]

       }

       

   }

}

sweep_id = wandb.sweep(sweep_config, project="hyperparameter-sweeps-object-detection-save-weights")
wandb.agent(sweep_id, function=train_sweep)