!pip install imgaug



# !git clone https://www.github.com/matterport/Mask_RCNN.git

import os

os.chdir('/kaggle/input/custom-maskrcnn/Mask_RCNN-master')

!python setup.py -q install



!pip uninstall keras -y

!pip install keras==2.2.1
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

print(tf.__version__)

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

    raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))



import os

import sys

import json

import datetime

import numpy as np

import skimage.draw

from matplotlib import pyplot as plt





# Root directory of the project

ROOT_DIR = '/kaggle/working'



# Import Mask RCNN



sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config

from mrcnn import model as modellib, utils

from mrcnn import visualize

import imgaug.augmenters as iaa
# Directory to save logs and model checkpoints, if not provided

# through the command line argument --logs

# DEFAULT_LOGS_DIR = os.path.join(r"/kaggle/working", "saved_checkpoints")

# os.mkdir(DEFAULT_LOGS_DIR)



# Path to trained weights file

COCO_WEIGHTS_PATH = os.path.join("/kaggle/input/datasetphase3", "mask_rcnn_defect_detection_latest.h5")

DATASET_PATH = r"/kaggle/input/datasetphase3"
import os

DEFAULT_LOGS_DIR = os.path.join(r"/kaggle/working", "logs")

os.mkdir(DEFAULT_LOGS_DIR)
############################################################

#  Configurations

############################################################

class TableConfig(Config):



    """Configuration for training on the toy dataset.

    Derives from the base Config class and overrides some values.

    """

    # Give the configuration a recognizable name

    NAME = "table_identification"

    # We use a GPU with 12GB memory, which can fit two images.

    # Adjust down if you use a smaller GPU.

    IMAGES_PER_GPU = 1

    # Number of classes (including background)

    NUM_CLASSES = 1 + 2  # Background + objects

    # Number of training steps per epoch

    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence

    DETECTION_MIN_CONFIDENCE = 0.9
# ############################################################



# #  Dataset



# ############################################################





# class TableDataset(utils.Dataset):



#     def load_VIA(self, dataset_dir, subset, hc=False):

#         """Load the surgery dataset from VIA.

#         dataset_dir: Root directory of the dataset.

#         subset: Subset to load: train or val or predict

#         """

#         # Add classes. We have only one class to add.

#         self.add_class("table_identification", 1, "yellow")

#         self.add_class("table_identification", 2, "red")

        

#         # Train or validation dataset?

#         assert subset in ["train", "val"]

#         dataset_dir = os.path.join(dataset_dir, subset)

#         annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))

#         annotations = list(annotations.values())  

#         # don't need the dict keys

#         # The VIA tool saves images in the JSON even if they don't have any

#         # annotations. Skip unannotated images.

#         annotations = [a for a in annotations if a['regions']]

#         # Add images

#         for a in annotations:

#             # Get the x, y coordinaets of points of the polygons that make up

#             # the outline of each object instance. There are stores in the

#             # shape_attributes (see json format above)

#             polygons = [r['shape_attributes'] for r in a['regions'].values()]

#             names = [r['region_attributes'] for r in a['regions'].values()]

#             # load_mask() needs the image size to convert polygons to masks.

#             # Unfortunately, VIA doesn't include it in JSON, so we must read

#             # the image. This is only managable since the dataset is tiny.

#             image_path = os.path.join(dataset_dir, a['filename'])

#             image = skimage.io.imread(image_path)

#             height, width = image.shape[:2]

#             self.add_image(

#                 "table_identification",

#                 image_id=a['filename'],  # use file name as a unique image id

#                 path=image_path,

#                 width=width, height=height,

#                 polygons=polygons,

#                 names=names)







#     def load_mask(self, image_id):

#         """Generate instance masks for an image.

#        Returns:

#         masks: A bool array of shape [height, width, instance count] with

#             one mask per instance.

#         class_ids: a 1D array of class IDs of the instance masks.

#         """

#         # If not a surgery dataset image, delegate to parent class.

#         image_info = self.image_info[image_id]

#         if image_info["source"] != "table_identification":

#             return super(self.__class__, self).load_mask(image_id)



#         # Convert polygons to a bitmap mask of shape

#         # [height, width, instance_count]

#         info = self.image_info[image_id]

#         class_names = info["names"]

#         mask = np.zeros([info["height"], info["width"], len(info["polygons"])],

#                         dtype=np.uint8)

#         for i, p in enumerate(info["polygons"]):

#             # Get indexes of pixels inside the polygon and set them to 1

#             rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

#             mask[rr, cc, i] = 1

#         # Assign class_ids by reading class_names

#         class_ids = np.zeros([len(info["polygons"])])

#         # In the surgery dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.

#         for i, p in enumerate(class_names):

#         #"name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}

#             if p['name'] == 'yellow':

#                 class_ids[i] = 1

#             elif p['name'] == 'red':

#                 class_ids[i] = 2

#             #assert code here to extend to other labels

#         class_ids = class_ids.astype(int)

#         # Return mask, and array of class IDs of each instance. Since we have

#         # one class ID only, we return an array of 1s

#         return mask.astype(np.bool), class_ids







#     def image_reference(self, image_id):

#         """Return the path of the image."""

#         info = self.image_info[image_id]

#         if info["source"] == "table_identification":

#             return info["path"]

#         else:

#             super(self.__class__, self).image_reference(image_id)







#     def load_mask_hc(self, image_id):

#         """Generate instance masks for an image.

#        Returns:

#         masks: A bool array of shape [height, width, instance count] with

#             one mask per instance.

#         class_ids: a 1D array of class IDs of the instance masks.

#         """

#         # If not a surgery dataset image, delegate to parent class.

#         image_info = self.image_info[image_id]

#         if image_info["source"] != "table_identification":

#             return super(self.__class__, self).load_mask(image_id)



#         # Convert polygons to a bitmap mask of shape

#         # [height, width, instance_count]

#         info = self.image_info[image_id]

#         #"name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}

#         class_names = info["names"]

#         mask = np.zeros([info["height"], info["width"], len(info["polygons"])],

#                         dtype=np.uint8)

#         for i, p in enumerate(info["polygons"]):

#             # Get indexes of pixels inside the polygon and set them to 1

#             rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

#             mask[rr, cc, i] = 1

#         # Assign class_ids by reading class_names

#         class_ids = np.zeros([len(info["polygons"])])

#         # In the surgery dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.

#         for i, p in enumerate(class_names):

#             if p['name'] == 'yellow':

#                 class_ids[i] = 14

#             elif p['name'] == 'red':

#                 class_ids[i] = 24           

#             else:

#                 class_ids[i] = int(p['name'])

#             #assert code here to extend to other labels

#         class_ids = class_ids.astype(int)

#         # Return mask, and array of class IDs of each instance. Since we have

#         # one class ID only, we return an array of 1s

#         return mask.astype(np.bool), class_ids





# def train(model, *dic):



#     """Train the model."""

#     # Training dataset.

#     dataset_train = TableDataset()

#     dataset_train.load_VIA(DATASET_PATH, "train")

#     dataset_train.prepare()

#     # Validation dataset

#     dataset_val = TableDataset()

#     dataset_val.load_VIA(DATASET_PATH, "val")

#     dataset_val.prepare()

#     # *** This training schedu le is an example. Update to your needs ***

#     # Since we're using a very small dataset, and starting from

#     # COCO trained weights, we don't need to train too long. Also,

#     # no need to train all layers, just the heads should do it.

#     print("Training network heads")

#     seq = iaa.Sequential([

#         iaa.Fliplr(0.5),

#         iaa.Flipud(0.5)

#         ])

    

#     # create checkpoint callback

#     cp_callback = [tf.keras.callbacks.ModelCheckpoint(DEFAULT_LOGS_DIR,

#                                                     save_best_only=True,

#                                                     save_weights_only=True,

#                                                     verbose=1)]

#     model.train(dataset_train, dataset_val,

#                 learning_rate=config.LEARNING_RATE,

#                 epochs=50000,

#                 layers='heads',

#                 augmentation = seq)#, custom_callbacks=cp_callback)



# def color_splash(image, mask):



#     """Apply color splash effect.

#     image: RGB image [height, width, 3]

#     mask: instance segmentation mask [height, width, instance count]

#     Returns result image.

#     """

#     # Make a grayscale copy of the image. The grayscale copy still

#     # has 3 RGB channels, though.

#     gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

#     # We're treating all instances as one, so collapse the mask into one layer

#     mask = (np.sum(mask, -1, keepdims=True) >= 1)

#     # Copy color pixels from the original color image where mask is set

#     if mask.shape[0] > 0:

#         splash = np.where(mask, image, gray).astype(np.uint8)

#     else:

#         splash = gray

#     return splash





# def detect_and_color_splash(model, image_path=None, video_path=None, out_dir=''):

#     assert image_path or video_path

#     class_names = ['BG', 'yellow', 'red']

#     # Image or video?

#     if image_path:

#         # Run model detection and generate the color splash effect

#         print("Running on {}".format(image_path))

#         # Read image

#         image = skimage.io.imread(image_path)

#         # Detect objects

#         r = model.detect([image], verbose=1)[0]

#         print(r)

#         # Color splash

#         splash = color_splash(image, r['masks'])

# #         visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],

# #                                     class_names, r['scores'], making_image=True)



#         file_name = 'splash.png'

#         # Save output

#         file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())

#         save_file_name = os.path.join(out_dir, file_name)

#         skimage.io.imsave(save_file_name, splash)

#     elif video_path:

#         import cv2

#         # Video capture

#         vcapture = cv2.VideoCapture(video_path)

#         # width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))

#         # height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         width = 1600

#         height = 1600

#         fps = vcapture.get(cv2.CAP_PROP_FPS)

#         # Define codec and create video writer

#         file_name = "splash_{:%Y%m%dT%H%M%S}.wmv".format(datetime.datetime.now())

#         vwriter = cv2.VideoWriter(file_name,

#                                   cv2.VideoWriter_fourcc(*'MJPG'),

#                                   fps, (width, height))



#         count = 0

#         success = True

#         #For video, we wish classes keep the same mask in frames, generate colors for masks

#         colors = visualize.random_colors(len(class_names))

#         while success:

#             print("frame: ", count)

#             # Read next image

#             plt.clf()

#             plt.close()

#             success, image = vcapture.read()

#             if success:

#                 # OpenCV returns images as BGR, convert to RGB

#                 image = image[..., ::-1]

#                 # Detect objects

#                 r = model.detect([image], verbose=0)[0]

#                 # Color splash

#                 # splash = color_splash(image, r['masks'])

#                 splash = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],

#                                                      class_names, r['scores'], colors=colors, making_video=True)

#                 # Add image to video writer

#                 vwriter.write(splash)

#                 count += 1

#         vwriter.release()

#     print("Saved to ", file_name)



# ############################################################

# #  RLE Encoding

# ############################################################



# def rle_encode(mask):

#     """Encodes a mask in Run Length Encoding (RLE).

#     Returns a string of space-separated values.

#     """

#     assert mask.ndim == 2, "Mask must be of shape [Height, Width]"

#     # Flatten it column wise

#     m = mask.T.flatten()

#     # Compute gradient. Equals 1 or -1 at transition points

#     g = np.diff(np.concatenate([[0], m, [0]]), n=1)

#     # 1-based indicies of transition points (where gradient != 0)

#     rle = np.where(g != 0)[0].reshape([-1, 2]) + 1

#     # Convert second index in each pair to lenth

#     rle[:, 1] = rle[:, 1] - rle[:, 0]

#     return " ".join(map(str, rle.flatten()))





# def rle_decode(rle, shape):

#     """Decodes an RLE encoded list of space separated

#     numbers and returns a binary mask."""

#     rle = list(map(int, rle.split()))

#     rle = np.array(rle, dtype=np.int32).reshape([-1, 2])

#     rle[:, 1] += rle[:, 0]

#     rle -= 1

#     mask = np.zeros([shape[0] * shape[1]], np.bool)

#     for s, e in rle:

#         assert 0 <= s < mask.shape[0]

#         assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)

#         mask[s:e] = 1

#     # Reshape and transpose

#     mask = mask.reshape([shape[1], shape[0]]).T

#     return mask



# def mask_to_rle(image_id, mask, scores):

#     "Encodes instance masks to submission format."

#     assert mask.ndim == 3, "Mask must be [H, W, count]"

#     # If mask is empty, return line with image ID only

#     if mask.shape[-1] == 0:

#         return "{},".format(image_id)

#     # Remove mask overlaps

#     # Multiply each instance mask by its score order

#     # then take the maximum across the last dimension

#     order = np.argsort(scores)[::-1] + 1  # 1-based descending

#     mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)

#     # Loop over instance masks

#     lines = []

#     for o in order:

#         m = np.where(mask == o, 1, 0)

#         # Skip if empty

#         if m.sum() == 0.0:

#             continue

#         rle = rle_encode(m)

#         lines.append("{}, {}".format(image_id, rle))

#     return "\n".join(lines)



# def detect(model, dataset_dir, subset):

#     """Run detection on images in the given directory."""

#     print("Running on {}".format(dataset_dir))

#     os.makedirs('RESULTS')

#     submit_dir = os.path.join(os.getcwd(), "RESULTS/")

#     # Read dataset

#     dataset = TableDataset()

#     dataset.load_VIA(dataset_dir, subset)

#     dataset.prepare()

#     # Load over images

#     submission = []

#     for image_id in dataset.image_ids:

#         # Load image and run detection

#         image = dataset.load_image(image_id)

#         # Detect objects

#         r = model.detect([image], verbose=0)[0]

#         # Encode image to RLE. Returns a string of multiple lines

#         source_id = dataset.image_info[image_id]["id"]

#         rle = mask_to_rle(source_id, r["masks"], r["scores"])

#         submission.append(rle)

#         # Save image with masks

#         canvas = visualize.display_instances(

#             image, r['rois'], r['masks'], r['class_ids'],

#             dataset.class_names, r['scores'], detect=True)

#             # show_bbox=False, show_mask=False,

#             # title="Predictions",

#             # detect=True)

#         canvas.print_figure("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"][:-4]))

#     # Save to csv file

#     submission = "ImageId,EncodedPixels\n" + "\n".join(submission)

#     file_path = os.path.join(submit_dir, "submit.csv")

#     with open(file_path, "w") as f:

#         f.write(submission)

#     print("Saved to ", submit_dir)############################################################



# #  Training



# ############################################################



# if __name__ == '__main__':



#     import argparse

#     # Parse command line arguments

# #     parser = argparse.ArgumentParser(

# #         description='Train Mask R-CNN to detect rings and robot arms.')

# #     parser.add_argument("command",

# #                         metavar="<command>",

# #                         help="'train' or 'splash'")

# #     parser.add_argument('--dataset', required=False,

# #                         metavar="/home/simon/mask_rcnn/data/surgery",

# #                         help='Directory of the surgery dataset')

# #     parser.add_argument('--weights', required=True,

# #                         metavar="/home/simon/logs/weights.h5",

# #                         help="Path to weights .h5 file or 'coco'")

# #     parser.add_argument('--logs', required=False,

# #                         default=DEFAULT_LOGS_DIR,

# #                         metavar="/path/to/logs/",

# #                         help='Logs and checkpoints directory (default=logs/)')

# #     parser.add_argument('--image', required=False,

# #                         metavar="path or URL to image",

# #                         help='Image to apply the color splash effect on')

# #     parser.add_argument('--video', required=False,

# #                         metavar="path or URL to video",

# #                         help='Video to apply the color splash effect on')

# #     parser.add_argument('--subset', required=False,

# #                         metavar="Dataset sub-directory",

# #                         help="Subset of dataset to run prediction on")

# #     args = parser.parse_args()

# #     # Validate arguments

# #     if args.command == "train":

# #         assert args.dataset, "Argument --dataset is required for training"

# #     elif args.command == "splash":

# #         assert args.image or args.video,\

# #                "Provide --image or --video to apply color splash"





# #     # Configurations

# #     if args.command == "train":

#     config = TableConfig()

# #     else:

# #         class InferenceConfig(SurgeryConfig):

# #             # Set batch size to 1 since we'll be running inference on

# #             # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

# #             GPU_COUNT = 1

# #             IMAGES_PER_GPU = 1

# #         config = InferenceConfig()

#     config.display()

#     # Create model

# #     if args.command == "train":

#     model = modellib.MaskRCNN(mode="training", config=config,

#                                   model_dir=DEFAULT_LOGS_DIR)

# #     else:

# #         model = modellib.MaskRCNN(mode="inference", config=config,

# #                                   model_dir=args.logs)

#     # Select weights file to load

# #     if args.weights.lower() == "coco":

#     weights_path = COCO_WEIGHTS_PATH

#         # Download weights file

# #         if not os.path.exists(weights_path):

# #             utils.download_trained_weights(weights_path)

# #     elif args.weights.lower() == "last":

# #         # Find last trained weights

# #         weights_path = model.find_last()[1]

# #     elif args.weights.lower() == "imagenet":

# #         # Start from ImageNet trained weights

# #         weights_path = model.get_imagenet_weights()

# #     else:

# #         weights_path = args.weights

#     # Load weights

#     print("Loading weights ", weights_path)

# #     if args.weights.lower() == "coco":

#         # Exclude the last layers because they require a matching

#         # number of classes

#     model.load_weights(weights_path, by_name=True, exclude=[

#             "mrcnn_class_logits", "mrcnn_bbox_fc",

#             "mrcnn_bbox", "mrcnn_mask"])

# #     else:

# #         model.load_weights(weights_path, by_name=True)



#     # Train or evaluate

# #     if args.command == "train":

#     print("Starting Training")

    

#     train(model)

    

# #     elif args.command == "detect":

# #         detect(model, args.dataset, args.subset)

# #     elif args.command == "splash":

# #         detect_and_color_splash(model, image_path=args.image,

# #                                 video_path=args.video)

# #     else:

# #         print("'{}' is not recognized. "

# #               "Use 'train' or 'splash'".format(args.command))



# # dataset_dir = '/home/simon/deeplearning/mask_rcnn/data'



# # dataset_train = SurgeryDataset()



# # dataset_train.VIA(dataset_dir, "train")



# # # dataset_train.prepare()



# # a, b = dataset_train.load_mask(130)



# # print(a.shape, b.shape)



# # print(b)