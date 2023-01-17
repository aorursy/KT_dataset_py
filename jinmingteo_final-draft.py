!conda install tensorflow-gpu==1.14.0 -y
# !pip install tensorflow-gpu==1.15
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
!pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

# from shutil import copy
# copy our file into the working directory (make sure it has .py suffix)
# copy(src = "/kaggle/input/til2020-test/ssd_tensorflow2/ssd_tensorflow2/", dst = "../working/ssd_tensorflow2/")
!cp -r /kaggle/input/til2020-test/ssd_tensorflow/ssd_tensorflow/* ./

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
%matplotlib inline
# Split train data into training data and validation data
val_split = 0.2

import json
import random
# read file
with open('../input/til2020-test/train.json', 'r') as myfile:
    data=myfile.read()

# parse file
train_ann = json.loads(data)
val_ann = {'annotations':[], 'images':[], 'categories':train_ann['categories']}
val_ann['images'] = random.sample(train_ann['images'], int(val_split*len(train_ann['images'])))

image_id_list =[]
for image in val_ann['images']:
    train_ann['images'].remove(image)
    image_id_list.append(image['id'])

for ann in train_ann['annotations']:
    if ann['image_id'] in image_id_list:
        val_ann['annotations'].append(ann)
        train_ann['annotations'].remove(ann)

with open('./train_split.json', 'w') as f:
    json.dump(train_ann, f)
    
with open('./val_split.json', 'w') as f:
    json.dump(val_ann, f)
img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 11 # Number of classes
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
K.clear_session()
import keras
# 1: Build the Keras model.
model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)

# 2: Load some weights into the model.
weights_path = '../input/til2020-test/ssd_tensorflow/ssd_tensorflow/VGG_ILSVRC_16_layers_fc_reduced.h5'

model.load_weights(weights_path, by_name=True)

# 3: Instantiate an optimizer and the SSD loss function and compile the model.
#    If you want to follow the original Caffe implementation, use the preset SGD
#    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)
# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.
train_dataset.parse_json(images_dirs=['../input/til2020-test/train/train/'],
                        annotations_filenames=['train_split.json'],
                        ground_truth_available=True)

val_dataset.parse_json(images_dirs=['../input/til2020-test/train/train/'],
                        annotations_filenames=['val_split.json'],
                        ground_truth_available=True)
# 3: Set the batch size.
batch_size = 32 # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))
# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001
    
# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='ssd300_epoch-{epoch:02d}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
#model_checkpoint.best = 

csv_logger = CSVLogger(filename='ssd300_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = 1
#steps_per_epoch = 1000

history = model.fit(train_generator,
                              #steps_per_epoch=ceil(train_dataset_size/batch_size),
                              steps_per_epoch=10,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)
from eval_utils.coco_utils import get_coco_category_maps, predict_all_to_json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
test_dir = '../input/til2020-test/test/test/'
test_ann = '../input/til2020-test/test.json'

test_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
test_dataset.parse_json(images_dirs=[test_dir],
                        annotations_filenames=[test_ann],
                        ground_truth_available=True,
                        include_classes='all')

# dataset = DataGenerator()


# dataset.parse_json(images_dirs=[test_dir],
#                    annotations_filenames=[test_ann],
#                    ground_truth_available=False, # It doesn't matter whether you set this `True` or `False` because the ground truth won't be used anyway, but the parsing goes faster if you don't load the ground truth.
#                    include_classes='all',
#                    ret=False)

# We need the `classes_to_cats` dictionary. Read the documentation of this function to understand why.
cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(test_ann)
from eval_utils.average_precision_evaluator import Evaluator
evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=test_dataset,
                      model_mode='inference')

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results
# results_file = 'ssd300_results.json'
# batch_size = 32
# predict_all_to_json(out_file=results_file,
#                     model=model,
#                     img_height=img_height,
#                     img_width=img_width,
#                     classes_to_cats=classes_to_cats,
#                     data_generator=test_dataset,
#                     batch_size=batch_size,
#                     data_generator_mode='resize',
#                     model_mode='inference',
#                     confidence_thresh=0.01,
#                     iou_threshold=0.45,
#                     top_k=200,
#                     normalize_coords=True)
# coco_gt   = COCO(test_ann)
# coco_dt   = coco_gt.loadRes(results_file)
# image_ids = sorted(coco_gt.getImgIds())
# cocoEval = COCOeval(cocoGt=coco_gt,
#                     cocoDt=coco_dt,
#                     iouType='bbox')
# cocoEval.params.imgIds  = image_ids
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
!pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

coco_gt   = COCO(test_ann)
coco_dt   = coco_gt.loadRes(results_file)
cocoEval = COCOeval(cocoGt=coco_gt,
                    cocoDt=coco_dt,
                    iouType='bbox')