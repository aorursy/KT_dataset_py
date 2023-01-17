!cp -r ../input/mask-rcnn/Mask_RCNN-master/* ./
import numpy as np

import pandas as pd

import pydicom

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import os

import sys

import math

from tqdm import tqdm

import seaborn as sns

import random



import cv2

from mrcnn.config import Config

from mrcnn import model as modellib

from mrcnn import visualize

from mrcnn import utils

from mrcnn.model import log



import tensorflow as tf

import keras
tf.__version__
tf.test.is_gpu_available()
dirname = '../input/rsna-pneumonia-detection-challenge/'

train = pd.read_csv(dirname+'stage_2_train_labels.csv')

#train.fillna(value=-1, inplace=True)

print(train.head())

print(train.describe())

#print(train.isnull())
pneumonia_locations = {}

train_images = []

target = []

for i in range(len(train)):

    n = str(dirname + 'stage_2_train_images/' + train['patientId'][i] + '.dcm')

    if n not in train_images:

        train_images.append(n)

        target.append(train['Target'][i])

    if train['Target'][i]==1:

        loc = [int(train['x'][i]), int(train['y'][i]),

               int(train['width'][i]), int(train['height'][i])]

        if n in pneumonia_locations:

            pneumonia_locations[n].append(loc)

        else:

            pneumonia_locations[n] = [loc]

train_images = pd.DataFrame(train_images)

train_images.columns = ['filepath']

train_images['Target'] = target
del train

print(train_images.head())

len(train_images)
X = list(train_images['filepath'])

random.seed(42)

random.shuffle(X)

l = int(0.2*len(X))

train_X = X[l:]

val_X =X[:l]
print(len(train_X))

print(len(val_X))
sns.countplot(train_images.Target)
i=0

ln = []

for item in pneumonia_locations:

    ln.append(len(pneumonia_locations[item]))

    i+=1

sns.countplot(ln)
%%time

im = (pydicom.dcmread(train_images['filepath'][0]).pixel_array)

plt.imshow(im)

plt.show()
fig = plt.figure(figsize = (15,10))

columns = 5

rows = 2

for i in [0,1]:

    df = train_images[train_images['Target']==i].sample(5)

    df = list(df['filepath'])

    for j in range(5):

        fig.add_subplot(rows, columns, i*columns + j +1)

        plt.imshow((pydicom.dcmread(df[j]).pixel_array)) #cmap = plt.cm.bone

    del df
f, axarr = plt.subplots(2, 5, figsize=(20, 15))

axarr = axarr.ravel()

axidx = 0

df1 = train_images[train_images['Target']==0].sample(5)

df2 = train_images[train_images['Target']==1].sample(5)

df = np.concatenate((np.array(df1['filepath']),np.array(df2['filepath'])),axis=0)

df = list(df)

for i in range(len(df)):

    axarr[axidx].imshow(pydicom.dcmread(df[i]).pixel_array)

    if df[i] in pneumonia_locations:

        l = pneumonia_locations[df[i]]

        for j in l:

            axarr[axidx].add_patch(patches.Rectangle((j[0], j[1]), j[2], j[3], linewidth=2, edgecolor='b', facecolor='none'))

    axidx+=1

plt.show()
class PneumoniaConfig(Config):

    NAME = 'Pneumonia'

    

    GPU_COUNT = 1

    IMAGES_PER_GPU = 8

    

    BACKBONE = 'resnet50'

    

    NUM_CLASSES = 2  # background + 1 pneumonia classes

    

    IMAGE_MIN_DIM = 256

    IMAGE_MAX_DIM = 256

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    TRAIN_ROIS_PER_IMAGE = 32

    MAX_GT_INSTANCES = 4

    DETECTION_MAX_INSTANCES = 3

    DETECTION_MIN_CONFIDENCE = 0.78  ## match target distribution

    DETECTION_NMS_THRESHOLD = 0.01



    STEPS_PER_EPOCH = 200



config = PneumoniaConfig()

config.display()
class DatasetGenerator(utils.Dataset):

    

    def __init__(self, fp, pneumonia_locations=None, batch_size=32, image_size=1024, predict=False):

        super().__init__(self)

        self.fp = fp

        self.pneumonia_locations = pneumonia_locations

        self.batch_size = batch_size

        self.image_size = image_size

        self.predict = predict

        

        self.add_class('Pneumonia', 1, 'pneumonia')

        for id, fps in enumerate(fp):

            self.add_image('Pneumonia',image_id=id,

                          path =fps)

        

    def load_mask(self, image_id):

        info = self.image_info[image_id]

        fps = info['path']

        if fps in self.pneumonia_locations:

            l = self.pneumonia_locations[fps]

            mask = np.zeros((self.image_size, self.image_size, len(l)), dtype=np.uint8)

            class_ids = np.zeros((len(l),), dtype=np.int32)

            i=0

            for a in l:

                x=int(a[0])

                y=int(a[1])

                w=int(a[2])

                h=int(a[3])

                mask_instance = mask[:,:,i].copy()

                cv2.rectangle(mask_instance, (x,y), (x+w,y+h), 255, -1)

                mask[:,:,i] = mask_instance

                class_ids[i] = 1

                i+=1

        else:

            mask = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)

            class_ids = np.zeros((1,), dtype=np.int32)

        return mask.astype(np.bool), class_ids.astype(np.int32)

    

    def load_image(self, image_id):

        info = self.image_info[image_id]

        fps = info['path']

        image = pydicom.dcmread(fps).pixel_array

        if len(image.shape) != 3 or image.shape[2] != 3:

            image = np.stack((image,) * 3, -1)

        return image

    

    def image_reference(self, image_id):

        info = self.image_info[image_id]

        return info['path']
train_gen = DatasetGenerator(train_X, pneumonia_locations)

train_gen.prepare()



val_gen = DatasetGenerator(val_X, pneumonia_locations)

val_gen.prepare()
# Load and display random samples

image_ids = np.random.choice(train_gen.image_ids,5)

for image_id in image_ids:

    image = train_gen.load_image(image_id)

    mask, class_ids = train_gen.load_mask(image_id)

    visualize.display_top_masks(image, mask, class_ids, 

                                train_gen.class_names, limit=1)

    print(mask.shape)

    print(image.shape)
# Load random image and mask.

image_id = np.random.choice(train_gen.image_ids, 1)[0]

image = train_gen.load_image(image_id)

mask, class_ids = train_gen.load_mask(image_id)



# Compute Bounding box

bbox = utils.extract_bboxes(mask)



# Display image and instances

visualize.display_instances(image, bbox, mask, class_ids, 

                            train_gen.class_names)
def model_definition():

    print("loading mask R-CNN model")

    model = modellib.MaskRCNN(mode='training', 

                              config=config, 

                              model_dir='/kaggle/working')

    

    # load the weights for COCO

    model.load_weights('/kaggle/input' + '/cocowg/mask_rcnn_coco.h5',

                       by_name=True, 

                       exclude=["mrcnn_class_logits",

                                "mrcnn_bbox_fc",  

                                "mrcnn_bbox","mrcnn_mask"])

    return model   



model = model_definition()
from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, CSVLogger)



def callback():

    cb = []

    checkpoint = ModelCheckpoint('/kaggle/working'+'/pneumonia_wg.h5',

                                 save_best_only=True,

                                 mode='min',

                                 monitor='val_loss',

                                 save_weights_only=True, verbose=1)

    cb.append(checkpoint)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',

                                   factor=0.3, patience=5,

                                   verbose=1, mode='auto',

                                   epsilon=0.0001, cooldown=1, min_lr=0.00001)

    log = CSVLogger('/kaggle/working'+'/pneumonia_history.csv')

    cb.append(log)

    cb.append(reduceLROnPlat)

    return cb
import warnings 

warnings.filterwarnings("ignore")
CB = callback()

LEARNING_RATE = 0.006



model.train(train_gen, val_gen, 

                learning_rate=LEARNING_RATE*2,

                custom_callbacks = CB,

                epochs=2, layers='all') 
history = model.keras_model.history.history
CB = callback()

LEARNING_RATE = 0.006



model.train(train_gen, val_gen, 

                learning_rate=LEARNING_RATE,

                custom_callbacks = CB,

                epochs=6, layers='heads') 
new_history = model.keras_model.history.history

for k in new_history: history[k] = history[k] + new_history[k]
# CB = callback()

# LEARNING_RATE = 0.006



# model.train(train_gen, val_gen, 

#                 learning_rate=LEARNING_RATE,

#                 custom_callbacks = CB,

#                 epochs=18, layers='all')
# new_history = model.keras_model.history.history

# for k in new_history: history[k] = history[k] + new_history[k]
epochs = range(1,len(next(iter(history.values())))+1)

pd.DataFrame(history, index=epochs)
plt.figure(figsize=(17,5))



plt.subplot(131)

plt.plot(epochs, history["loss"], label="Train loss")

plt.plot(epochs, history["val_loss"], label="Valid loss")

plt.legend()

plt.subplot(132)

plt.plot(epochs, history["mrcnn_class_loss"], label="Train class ce")

plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid class ce")

plt.legend()

plt.subplot(133)

plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train box loss")

plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid box loss")

plt.legend()



plt.show()
class InferenceConfig(PneumoniaConfig):

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    

inferconfig = InferenceConfig()

model = modellib.MaskRCNN(mode='inference', 

                              config=inferconfig,

                              model_dir='/kaggle/working')

model.load_weights('/kaggle/working/pneumonia_wg.h5',

                  by_name = True)
test = []

for dname, _, filenames in os.walk('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images/'):

    for filename in filenames:

        test.append(str(os.path.join(dname, filename)))

print(test[:5])
def predict(fp, filepath='submission.csv', min_conf=0.95):

    with open(filepath, 'w') as file:

        resize_factor = 1024/ config.IMAGE_SHAPE[0]

        

        file.write("patientId,PredictionString\n")



        for image_id in tqdm(fp):

            image = pydicom.read_file(image_id).pixel_array

            # If grayscale. Convert to RGB for consistency.

            if len(image.shape) != 3 or image.shape[2] != 3:

                image = np.stack((image,) * 3, -1)

            image, window, scale, padding, crop = utils.resize_image(

                image,

                min_dim=config.IMAGE_MIN_DIM,

                min_scale=config.IMAGE_MIN_SCALE,

                max_dim=config.IMAGE_MAX_DIM,

                mode=config.IMAGE_RESIZE_MODE)



            patient_id = os.path.splitext(os.path.basename(image_id))[0]



            results = model.detect([image])

            r = results[0]



            out_str = ""

            out_str += patient_id

            out_str += ","

            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )

            if len(r['rois']) == 0:

                pass

            else:

                num_instances = len(r['rois'])



                for i in range(num_instances):

                    if r['scores'][i] > min_conf:

                        out_str += ' '

                        out_str += str(round(r['scores'][i], 2))

                        out_str += ' '



                        # x1, y1, width, height

                        x1 = r['rois'][i][1]

                        y1 = r['rois'][i][0]

                        width = r['rois'][i][3] - x1

                        height = r['rois'][i][2] - y1

                        bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \

                                                           width*resize_factor, height*resize_factor)

                        out_str += bboxes_str



            file.write(out_str+"\n")
submission = os.path.join('/kaggle/working', 'submission.csv')

predict(test, filepath=submission)
submit = pd.read_csv(submission)

submit.head(10)