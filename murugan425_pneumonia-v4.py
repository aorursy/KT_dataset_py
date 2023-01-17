import os

import random

import numpy as np

import pandas as pd

import seaborn as sb

import skimage.io as io

import cv2

import math
import matplotlib.pyplot as plt

import matplotlib.patches as patches

import matplotlib.gridspec as gspec
import warnings

warnings.filterwarnings("ignore")
import random

random.seed(0)
import tensorflow as tf

tf.__version__
import pydicom

import glob

import pylab
datasetDir = '../input/rsna-pneumonia-detection-challenge/'

#datasetDir = '/Users/murugan/Repository/learn-ml/Great Lakes/datasets/pneumonia/'
bbox_df = pd.read_csv(datasetDir + 'stage_2_train_labels.csv')

targets_df = pd.read_csv(datasetDir + 'stage_2_detailed_class_info.csv')
bbox_df.isnull().values.any(), targets_df.isnull().values.any()
bbox_df.head()
targets_df.head()
bbox_df.patientId.head()
bbox_df.sort_values('patientId')

targets_df.sort_values('patientId')
bbox_df.shape, targets_df.shape
bbox_df['patientId'].value_counts().shape[0]
bbox_df[bbox_df.Target == 1]['patientId'].value_counts().shape[0]
bbox_df[bbox_df.Target == 0]['patientId'].value_counts().shape[0]
bbox_df[bbox_df.Target == 0].shape
targets_df['patientId'].value_counts().shape[0]
bbox_df.count().T
bbox_df.isnull().sum()
bbox_w_targets_df = pd.concat([bbox_df, targets_df.drop('patientId', axis=1)], axis=1)

bbox_w_targets_df.head(10)
bbox_w_targets_df.groupby(['class', 'Target']).size().reset_index(name='Count By Class')
duplicateCounts_df = bbox_w_targets_df.groupby('patientId').size().reset_index(name='counts')

duplicateCounts_df.groupby('counts').size().reset_index(name='Count By Duplicates')
duplicateCounts_df[duplicateCounts_df.counts == 4].sample(3)
#Just refering a sample of one of the patient Id with 4 bounding boxes

bbox_w_targets_df[bbox_w_targets_df.patientId == 

                  duplicateCounts_df[duplicateCounts_df.counts == 4].iloc[0].patientId]
bbox_w_counts = pd.merge(bbox_w_targets_df, duplicateCounts_df, on='patientId')

bbox_w_counts.head()
print('Non Null Values Count: ', bbox_w_targets_df.x.notnull().sum())

print('Null Values Count: ', bbox_w_targets_df.x.isnull().sum())

print('Total Count: ', bbox_w_targets_df.x.notnull().sum() + bbox_w_targets_df.x.isnull().sum())

print("Null Values in % : {0} ({1:2.2f}%)".format(bbox_w_targets_df.x.isnull().sum(), 

                                                  (bbox_w_targets_df.x.isnull().sum()/len(bbox_w_targets_df))*100))
fig = plt.figure(figsize=(18,5))

fig.add_subplot(1, 3, 1)

p1 = sb.countplot(bbox_w_targets_df['Target'])

fig.add_subplot(1, 3, 2)

p2 = sb.countplot(bbox_w_targets_df['class'])

plt.show()
ax = sb.countplot(bbox_w_targets_df['class'])

ax.set(title = 'Class Distribution')

plt.show()
class_disc = bbox_w_targets_df['class'].value_counts()

print('Percentage of patients with No Long opacity/Not Normal : {:5d} or {:.2f}%'.format(class_disc[0],(class_disc[0]/bbox_w_targets_df['class'].count())*100))

print('Percentage of patients with Long opacity : {:5d} or {:.2f}% '.format(class_disc[1],(class_disc[1]/bbox_w_targets_df['class'].count())*100))

print('Percentage of patients with Normal : {:5d} or {:.2f}% '.format(class_disc[2],(class_disc[2]/bbox_w_targets_df['class'].count())*100))
dicom_img_dir = os.path.join(datasetDir, 'stage_2_train_images')

dicom_img_dir
filenames = os.listdir(dicom_img_dir)

len(filenames)
train_images_dir = os.path.join(datasetDir,'stage_2_train_images')

test_images_dir = os.path.join(datasetDir,'stage_2_test_images')
print('Total number of Training images available are : {:5d}'.format(len(list(glob.iglob(train_images_dir + "/*.dcm", recursive=True)))))

print('Total number of Test images available are : {:5d}'.format(len(list(glob.iglob(test_images_dir + "/*.dcm", recursive=False)))))
dcm_file = os.path.join(dicom_img_dir, filenames[0])

dcm_file
pydicom.read_file(dcm_file)
dcm_img = pydicom.read_file(dcm_file).pixel_array

dcm_img.shape
dcm_img
dcm_img_3ch = np.stack([dcm_img]*3, -1)

dcm_img_3ch.shape
fig = plt.figure(figsize=(20,20))

fig.add_subplot(1, 3, 1)

plt.imshow(dcm_img)

fig.add_subplot(1, 3, 2)

plt.imshow(dcm_img_3ch)

plt.show()
# Extract Bounding box from data sst and visualize the bounding box in image

def extract_data(dataset):

    extract_bbox = lambda row: [row['y'], row['x'], row['height'], row['width']]

    datacol = {}

    index = 0

    for n, row in dataset.iterrows():        

        pid = row['patientId']

        if pid not in datacol:

            index = index+1

            datacol[pid] = {

                'dicom': train_images_dir + '/%s.dcm' % pid,

                'label': row['Target'],

                'boxes': [],

                'index': index}

            

        if datacol[pid]['label'] == 1:

            datacol[pid]['boxes'].append(extract_bbox(row))

    return datacol
bbox_data_dict = extract_data(bbox_w_targets_df)

bbox_data_dict
#Select a patient Id with postive target value '1'

samplePatientId = bbox_df[bbox_df.Target == 1]['patientId'].iloc[0]

samplePatientId
def plotBoundingBoxes(imgdata):

    d = pydicom.read_file(imgdata['dicom'])

    im = d.pixel_array    

    im = np.stack([im] * 3, axis=2)

    #Add boxes with random color if present

    for box in imgdata['boxes']:

        rgb = np.floor(np.random.rand(3) * 256).astype('int')

        im = overlayBoudingBox(im=im, box=box, rgb=rgb, stroke=6)



    pylab.imshow(im, cmap=pylab.cm.gist_gray)

    pylab.axis('off')



def overlayBoudingBox(im, box, rgb, stroke=1):

    #Convert coordinates to integers

    box = [int(b) for b in box]

    

    #Extract coordinates

    y1, x1, height, width = box

    y2 = y1 + height

    x2 = x1 + width



    im[y1:y1 + stroke, x1:x2] = rgb

    im[y2:y2 + stroke, x1:x2] = rgb

    im[y1:y2, x1:x1 + stroke] = rgb

    im[y1:y2, x2:x2 + stroke] = rgb

    return im
plotBoundingBoxes(bbox_data_dict[samplePatientId])
imgdata = bbox_data_dict[samplePatientId]

d = pydicom.read_file(imgdata['dicom'])

im = d.pixel_array    

im = np.stack([im] * 3, axis=2)

for box in imgdata['boxes']:

    rgb = np.floor(np.random.rand(3) * 256).astype('int')

    im = overlayBoudingBox(im=im, box=box, rgb=rgb, stroke=6)

pylab.imshow(im, cmap=pylab.cm.gist_gray)

pylab.axis('off')

plt.show()
bbox_df[bbox_df.Target == 1].index
rand_positive_patientIds = random.sample(list(bbox_df[bbox_df.Target == 1].index), 1)

rand_positive_patientIds

for i in rand_positive_patientIds:

    print(bbox_df[bbox_df.index == i].patientId.iloc[0])
rand_positive_index = random.sample(list(bbox_df[bbox_df.Target == 1].index), 5)

fig = plt.figure()

fig.set_figheight(25)

fig.set_figwidth(25)

pltloc = 0

for randIndex in rand_positive_index:

    pltloc += 1

    patientId = bbox_df[bbox_df.index == randIndex].patientId.iloc[0]

    a = fig.add_subplot(1, 5, pltloc)

    a.set(title = 'Index:' + str(randIndex))

    plotBoundingBoxes(bbox_data_dict[patientId])

plt.show()
from tensorflow.keras.applications.mobilenet import preprocess_input
len(bbox_data_dict)
IMAGE_DIM = 256
for key, value in list(bbox_data_dict.items())[0:10]:

    print(key, value['dicom'], value['label'], value['boxes'], value['index'])
sample_count = math.ceil(10/100 * len(bbox_data_dict))

sample_count
ARRAY_DIM = int(sample_count+1)

#TODO: Will process only the first 50/100 records initially

#ARRAY_DIM = int(len(bbox_data_dict))

masks = np.zeros((ARRAY_DIM, IMAGE_DIM, IMAGE_DIM))

X = np.zeros((ARRAY_DIM, IMAGE_DIM, IMAGE_DIM, 3))

Y = np.zeros((ARRAY_DIM, IMAGE_DIM, IMAGE_DIM))
def displayProcessedImageAndMask(imageIndex):

    for key, value in list(bbox_data_dict.items())[(imageIndex-1):imageIndex]:

        #print(key, value['dicom'], value['label'], value['boxes'], value['index'])

        dcm_path = value['dicom']

        dcm_data = pydicom.read_file(dcm_path)

        img = dcm_data.pixel_array

        mask = np.zeros(img.shape)

        img = cv2.resize(img, dsize=(IMAGE_DIM, IMAGE_DIM), interpolation=cv2.INTER_CUBIC)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        fig = plt.figure(figsize=(20,20))

        ax1 = fig.add_subplot(1, 3, 1)

        ax1.set_title("Rezied Original Image")

        plt.imshow(img) 

        ax2 = fig.add_subplot(1, 3, 2)

        ax2.set_title("Processed Input Image")

        plt.imshow(preprocess_input(np.array(img, dtype=np.float32)))

        ax3 = fig.add_subplot(1, 3, 3)

        ax3.set_title("Mask Layer")

        for boxes in value['boxes']:

                x1, y1, w, h = boxes

                y2 = y1 + h

                x2 = x1 + w

                mask[int(x1):int(x2), int(y1):int(y2)] = 1

        plt.imshow(cv2.resize(mask, dsize=(IMAGE_DIM, IMAGE_DIM)))

    plt.show()
displayProcessedImageAndMask(5)
def generateData(n=10):

    for key, value in list(bbox_data_dict.items())[0:n]:

        #print(key, value['dicom'], value['label'], value['boxes'], value['index'])

        dcm_path = value['dicom']

        index = value['index']

        target = value['label']

        dcm_data = pydicom.read_file(dcm_path)

        img = dcm_data.pixel_array

        mask = np.zeros(img.shape)

        img = cv2.resize(img, dsize=(IMAGE_DIM, IMAGE_DIM), interpolation=cv2.INTER_CUBIC)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        X[index] = preprocess_input(np.array(img, dtype=np.float32))

        for boxes in value['boxes']:

            x1, y1, w, h = boxes

            y2 = y1 + h

            x2 = x1 + w

            mask[int(x1):int(x2), int(y1):int(y2)] = 1

        masks = cv2.resize(mask, dsize=(IMAGE_DIM, IMAGE_DIM))

        Y[index] = masks

    return X,Y
generateData(sample_count)

X.shape, Y.shape
#Split the data into training and validation dataset

split_index = 10

X_train = X[split_index:]

X_valid = X[:split_index]

Y_train = Y[split_index:]

Y_valid = Y[:split_index]

X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape
from tensorflow.keras.layers import Concatenate, UpSampling2D, Conv2D, Reshape, Conv2DTranspose, Dropout, Lambda

from tensorflow.keras.models import Model

from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
def iou_bce_loss(y_true, y_pred):

    return 0.5 * binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)
# mean iou as a metric

def mean_iou(y_true, y_pred):

    y_pred = tf.round(y_pred)    

    intersect = tf.reduce_sum(y_true * y_pred, axis=[1])

    union = tf.reduce_sum(y_true, axis=[1]) + tf.reduce_sum(y_pred, axis=[1])

    smooth = tf.ones(tf.shape(intersect))

    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
def loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) - log(dice_coefficient(y_true, y_pred) + epsilon())
def dice_coefficient(y_true, y_pred):

    numerator = 2 * tensorflow.reduce_sum(y_true * y_pred)

    denominator = tensorflow.reduce_sum(y_true + y_pred)

    return numerator / (denominator + epsilon())
# define iou or jaccard loss function

def iou_loss(y_true, y_pred):

    y_true = tf.reshape(y_true, [-1])

    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)

    return 1 - score
def create_model(trainable=True):

    model = MobileNet(input_shape=(IMAGE_DIM, IMAGE_DIM, 3), 

                      include_top=False, alpha=1, weights="imagenet")



    for layer in model.layers[:-20]:

        layer.trainable = trainable



    block1 = model.get_layer("conv1_relu").output

    block2 = model.get_layer("conv_pw_1_relu").output

    block3 = model.get_layer("conv_pw_3_relu").output

    block4 = model.get_layer("conv_pw_5_relu").output

    block5 = model.get_layer("conv_pw_11_relu").output

    block6 = model.get_layer("conv_pw_13_relu").output



    x = Concatenate()([UpSampling2D()(block6), block5])

    x = Conv2D(100, (1, 1), activation='relu') (x)

    x = Concatenate()([UpSampling2D()(x), block4])

    x = Conv2D(100, (1, 1), activation='relu') (x)

    x = Concatenate()([UpSampling2D()(x), block3])

    x = Conv2D(100, (1, 1), activation='relu') (x)

    x = Concatenate()([UpSampling2D()(x), block2])

    x = Conv2D(100, (1, 1), activation='relu') (x)

    x = Concatenate()([UpSampling2D()(x),UpSampling2D()(block1)])

    x = Conv2D(1, kernel_size=1,strides=1, activation="sigmoid")(x)

    x = Reshape((256, 256,1))(x)

    

    return Model(inputs=model.input, outputs=x)
# Give trainable=False as argument, if you want to freeze lower layers for fast training (but low accuracy)

model1 = create_model()



# Print summary

print(model1.summary())
#Print the layer details of mobilenet and improve the Upsampling

for layer in model1.layers:

    if('conv_pw' in layer.name and 'relu' in layer.name):

        print(layer.name, layer.output_shape)
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model1.compile(optimizer=optimizer, loss=iou_bce_loss,  metrics=['accuracy', mean_iou])
checkpoint = ModelCheckpoint("model-{loss:.2f}.h5", monitor="loss", verbose=1, save_best_only=True,

                             save_weights_only=True, mode="min", save_freq=1)

stop = EarlyStopping(monitor="loss", patience=5, mode="min")

reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="min")
history = model1.fit(X_train, Y_train, validation_data = (X_valid, Y_valid), 

           epochs=10, batch_size=32, verbose=1, callbacks=[checkpoint, stop, reduce_lr])
PRED_DIM = int(1)

X_test = np.zeros((PRED_DIM, IMAGE_DIM, IMAGE_DIM, 3))

Y_test = np.zeros((PRED_DIM, IMAGE_DIM, IMAGE_DIM))

mask_pred = np.zeros((PRED_DIM, IMAGE_DIM, IMAGE_DIM))

def viewModelPrediction(n, model=model1):

    for key, value in list(bbox_data_dict.items())[(n-1):n]:

        #print(key, value['dicom'], value['label'], value['boxes'], value['index'])

        dcm_path = value['dicom']

        actul_target = value['label']

        dcm_data = pydicom.read_file(dcm_path)

        img = dcm_data.pixel_array

        mask = np.zeros(img.shape)

        img = cv2.resize(img, dsize=(IMAGE_DIM, IMAGE_DIM), interpolation=cv2.INTER_CUBIC)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        X_test[0] = preprocess_input(np.array(img, dtype=np.float32))

        preds = model1.predict(X_test)

        fig = plt.figure(figsize=(20,20))

        ax1 = fig.add_subplot(1, 4, 1)

        ax1.set_title("Rezied Original Image")

        plt.imshow(img) 

        ax2 = fig.add_subplot(1, 4, 2)

        ax2.set_title("Processed Input Image")

        plt.imshow(preprocess_input(np.array(img, dtype=np.float32)))

        ax3 = fig.add_subplot(1, 4, 3)

        ax3.set_title("Mask Layer")

        for boxes in value['boxes']:

                x1, y1, w, h = boxes

                y2 = y1 + h

                x2 = x1 + w

                mask[int(x1):int(x2), int(y1):int(y2)] = 1

        plt.imshow(cv2.resize(mask, dsize=(IMAGE_DIM, IMAGE_DIM)))

        ax4 = fig.add_subplot(1, 4, 4)

        ax4.set_title("Predicted Mask Layer")

        mask_pred = preds[0]

        plt.imshow(cv2.resize(mask_pred, dsize=(IMAGE_DIM, IMAGE_DIM)))

    plt.show()

    

viewModelPrediction(5)
viewModelPrediction(8)
for key, value in list(bbox_data_dict.items())[0:50]:

    if(value['label'] == 1):

        viewModelPrediction(value['index'])