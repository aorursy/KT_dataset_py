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
#Importing all the standard libraries
#..... array/martrix operations and dataframe libraries
import numpy as np
import pandas as pd
#...........
#.......... Visulaization libraries
import pydicom
import pylab
import matplotlib.pyplot as plt
import seaborn as sn
from skimage.transform import resize

#......
from sklearn.model_selection import train_test_split

# NN model building linraries
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ReduceLROnPlateau
#...................................................
# setting path for each of the files
class_path='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv'
labels_path='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
Image_train_path='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'
# Files descrition
#1. stage_2_detailed_class_info.csv- contains the information of target label
#2. stage_2_train_labels.csv- contains information on Target and bounding box
#3. stage_2_train_images- contains training images in dcm format
# Reading class file (first file) as dataframe and check few entries and shape
df_class=pd.read_csv(class_path)
print(df_class.head(10))
print(df_class.shape[0])
df_class['class'].value_counts()
#Observation:
# This file ocntains patient Id and repective class ifnormation. 
#. There are 30277 records
# There are three classes- 
#    1. Lung Opacity- Patient havinig pneumonia, 
#    2. Normal- Patient not having pnemonia and not having any other lung problem
#    3. No Lung Opacity/Not Normal- Patient not having pnemonia but having any other lung problem
df_class.info()
# checking the number of unique entries with respect to patient ID
#print(df_class['patientId'].value_counts().shape[0],'patient cases')
# # Reading label file (second file) as dataframe and check few entries and shape
df_label=pd.read_csv(labels_path)
print(df_label.head())
print(df_label.shape)
# Now lets drop the duplicate cases
df=pd.concat([df_label,df_class.drop('patientId',1)],1)
print(df.shape)
print(df.head())
# Classes and Targets based on Patient count
df.groupby(['class','Target']).size().reset_index(name='patient_numbers')
#print('Number of duplicate entries accross rows:\n', df[df.duplicated()].count())
#print('Number of duplicate Patient Id entries :\n', df[df.duplicated(subset='patientId')].count())
#print('Number of unique Patient Id entries: \n', df['patientId'].nunique())
#print('Count of various classes: \n',df.groupby('class')['patientId'].nunique())
# Observation
#1. All the Normal and No Lung Opacity / Not Normal	patients are grouped under Target label 0 (no pnemonia)
#2. Data Imabalance- there are ~30% pneumonia records and rest ~70% no pneumonia
#3  There are no duplicates accross rows
#4. Checking for duplicate patientId's, there are 26684 unique Patient Ids
# chekcing the type of image file format and total number of images
image_path='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'
print(os.listdir(image_path)[0])
import glob
print(len(list(glob.iglob("/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/*.dcm", recursive=True))))
# Checking sample image file for first entry in dataframe which is normal case
print(df.iloc[3])
patientId = df['patientId'][3]
image_path_1='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' %patientId
dcm_data=pydicom.read_file(image_path_1)
print(dcm_data)
#size of image
dcm_data.pixel_array.shape
#Plotting the image 
plt.figure(figsize=(12,10))
plt.subplot(121)
plt.title('Pateint- Normal case class')
plt.imshow(dcm_data.pixel_array)
plt.subplot(122)
plt.title('Pateint- Normal case class')
plt.imshow(dcm_data.pixel_array,cmap=plt.cm.gist_gray)
#... Sample No Lung Opacity / Not Normal case ----------------
print(df.iloc[0])
patientId = df['patientId'][0]
image_path_1='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' %patientId
dcm_data=pydicom.read_file(image_path_1)
print(dcm_data)

#Plotting the image 
plt.figure(figsize=(12,10))
plt.subplot(121)
plt.title('Pateint- No Lung Opacity / Not Normal case')
plt.imshow(dcm_data.pixel_array)
plt.subplot(122)
plt.title('Pateint- No Lung Opacity / Not Normal case')
plt.imshow(dcm_data.pixel_array,cmap=plt.cm.gist_gray)
# Lets us plot one Patient with pnemonia (Target = 1)
print(df.iloc[4])
patientId = df['patientId'][4]
image_path_1='/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' %patientId
dcm_data=pydicom.read_file(image_path_1)
print(dcm_data)
#Plotting the image 
plt.figure(figsize=(12,10))
plt.subplot(121)
plt.title('Pateint- With pneumonia class')
plt.imshow(dcm_data.pixel_array)
plt.subplot(122)
plt.title('Pateint- With pneumonia class')
plt.imshow(dcm_data.pixel_array,cmap=plt.cm.gist_gray)
# Function to show to a sample image with overlayed bounding box 
def showImage(row):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    imagePath = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/{0}.dcm".format(row['patientId'])
    d = pydicom.read_file(imagePath)
    image = d.pixel_array
    image = np.stack([image] * 3, axis=2)

    if row['Target'] == 1:        
        image = drawbox(image=image, row=row)

    plt.imshow(image, cmap=plt.cm.gist_gray)
    
    
def drawbox(image, row):
    color = np.floor(np.random.rand(3) * 256).astype('int')
    stroke=6
  
    # --- Extract coordinates
    x1 = int(row['x'])
    y1 = int(row['y'])
    y2 = y1 + int(row['height'])
    x2 = x1 + int(row['width'])
    
    #print(x1)
    #print(x2)
    #print(y1)
    #print(y2)
    
    image[y1:y1 + stroke, x1:x2] = color
    image[y2:y2 + stroke, x1:x2] = color
    image[y1:y2, x1:x1 + stroke] = color
    image[y1:y2, x2:x2 + stroke] = color

    return image
#patient = labels_w_class.iloc[[10]]
patient = list(df.T.to_dict().values())[4]
print("Path : stage_2_train_images/{0}.dcm".format(patient['patientId']))
print("Target : {0}".format(patient['Target']))

plt.figure(figsize=(7,7))
plt.title("Sample Patient - Lung Opacity")
showImage(patient)
# Function to collect three major information (Patient ID, box and corresponding image file path) into a dictonary called parsed
extract_boxes=lambda row: [ row['y'], row['x'], row ['height'], row['width']]
parsed={}
for n, row in df.iterrows():
    pid=row['patientId']
    if pid not in parsed:
        parsed[pid]={
            'dicom': '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % pid,
            'label': row['Target'],
            'boxes':[]
            }
                 
    if parsed[pid]['label']==1:
        parsed[pid]['boxes'].append(extract_boxes(row))
len(parsed)
parsed[df['patientId'][4]]
imagePath = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/{0}.dcm".format(row['patientId'])
d = pydicom.read_file(imagePath)
image = d.pixel_array
image
patientId
#............ function to get the images with bounbding box for any given Patient ID
def draw(data):
#    """
#    Method to draw single patient with bounding box(es) if present 

#   """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    pylab.imshow(im, cmap=pylab.cm.gist_gray)
    pylab.axis('off')

def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im
# overalaping bounding box with image for sample pnemonia case
#Plotting the image 
print(df.iloc[4])
patientId = df['patientId'][4]
draw(parsed[patientId])
import csv
# empty dictionary
pneumonia_locations = {}
# load table
with open(os.path.join(labels_path), mode='r') as infile:
    # open reader
    reader = csv.reader(infile)
    # skip header
    next(reader, None)
    # loop through rows
    for rows in reader:
        # retrieve information
        filename = rows[0]
        location = rows[1:5]
        pneumonia = rows[5]
        # if row contains pneumonia add label to dictionary
        # which contains a list of pneumonia locations per filename
        if pneumonia == '1':
            # convert string to float to int
            location = [int(float(i)) for i in location]
            # save pneumonia location in dictionary
            if filename in pneumonia_locations:
                pneumonia_locations[filename].append(location)
            else:
                pneumonia_locations[filename] = [location]
pneumonia_locations.values()
# load and shuffle filenames
folder = Image_train_path
filenames = os.listdir(folder)

#valid_filenames= os.listdir(folder)
from skimage.transform import resize
import keras
import random
class generator(keras.utils.Sequence):
    
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32, image_size=320, shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        if filename in self.pneumonia_locations:
            # loop through pneumonia
            for location in self.pneumonia_locations[filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h = location
                msk[y:y+h, x:x+w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)
image
img_base = pydicom.dcmread('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm').pixel_array
# create empty mask
msk = np.zeros(img_base.shape)
# get filename without extension
#filename = filename.split('.')[0]
# if image contains pneumonia
x, y, w, h = pneumonia_locations['00436515-870c-4b36-a041-de91049b9ab4'][0]
msk[y:y+h, x:x+w] = 1
# resize both image and mask
img = resize(img_base, (320, 320), mode='reflect')
msk = resize(msk, (320, 320), mode='reflect') > 0.5
# add trailing channel dimension
#img = np.expand_dims(img, -1)
#msk = np.expand_dims(msk, -1)
print(img_base)
print(img)
pneumonia_locations['00436515-870c-4b36-a041-de91049b9ab4'][0]
len(train_gen[0])
def create_downsample(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=3, depth=5):
    # input
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    #residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(256, 1, activation=None)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2DTranspose(128,8,strides=(4,4), padding="same", activation=None)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = keras.layers.UpSampling2D(2**(depth-2))(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

import tensorflow as tf
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

# cosine learning rate annealing
def cosine_annealing(x):
    lr = 0.001
    epochs = 1
    return lr*(np.cos(np.pi*x/epochs)+1.)/2
learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)
from keras.optimizers import Adam
# create network and compiler
model = create_network(input_size=320, channels=32, n_blocks=3, depth=5)
#model.load_weights('/kaggle/input/model-1/model_val-0.48.h5',)
learning_rate=0.005
decay=learning_rate/3.0
opt = Adam(lr = learning_rate)
model.compile(optimizer=opt,
              loss=iou_bce_loss,
              metrics=['accuracy', mean_iou])
model.summary()
# load and shuffle filenames
filenames = os.listdir(Image_train_path)
random.shuffle(filenames)
# split into train and validation filenames
n_valid_samples = 2560
train_filenames = filenames[n_valid_samples:]
valid_filenames = filenames[:n_valid_samples]
#print('n train samples', len(train_filenames))
#print('n valid samples', len(valid_filenames))
n_train_samples = len(filenames) - n_valid_samples
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("model_val-{loss:.2f}.h5", monitor="loss", verbose=1, save_best_only=True,
                             save_weights_only=True, mode="max", period=1) # Checkpoint best validation model
#stop = EarlyStopping(monitor="loss", patience=PATIENCE, mode="max") # Stop early, if the validation error deteriorates
#reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")

# create train and validation generators
#folder = '../input/stage_1_train_images'
train_gen = generator(folder, train_filenames, pneumonia_locations, batch_size=32, image_size=320, shuffle=True, augment=True, predict=False)
valid_gen = generator(folder, valid_filenames, pneumonia_locations, batch_size=32, image_size=320, shuffle=False, predict=False)

#history = model.fit_generator(train_gen,validation_data=valid_gen, callbacks=[checkpoint], epochs=3, workers=4, use_multiprocessing=True)

# save dictionary as csv file
sub = pd.DataFrame.from_dict(submission_dict,orient='index')
sub.index.names = ['patientId']
sub.columns = ['PredictionString']
sub.to_csv('submission_val.csv')
