import warnings; warnings.filterwarnings('ignore')

import pandas as pd, numpy as np

import matplotlib.pyplot as plt



import os, zipfile, random, csv

import seaborn as sns

import pydicom as dcm

from glob import glob

import cv2
# # Install packages

#!pip install -q pydicom


print(f'Current working directory: {os.getcwd()}')

print('Folder and Files in current directory: {}'.format(os.listdir()))

PATH = '/kaggle/'

DATA_FOLDER = os.path.join(PATH, 'input/rsna-pneumonia-detection-challenge/')

Folder_DCM_IMAGES = os.path.join(DATA_FOLDER,'stage_2_train_images/')



WORKING_FOLDER = os.path.join(PATH,'working/')

SAVE_PATH = os.path.join(WORKING_FOLDER,'Saved_Data/')



if not os.path.exists(SAVE_PATH):

    os.makedirs(SAVE_PATH)

    



print("DATA_FOLDER: ", DATA_FOLDER)  

print("Folder_DCM_IMAGES: ", Folder_DCM_IMAGES) 

print("SAVE_PATH: ", SAVE_PATH) 

print("WORKING_FOLDER: ", WORKING_FOLDER) 



os.chdir(WORKING_FOLDER)

print(f'Current working directory: {os.getcwd()}')
train_labels = pd.read_csv(os.path.join(DATA_FOLDER,'stage_2_train_labels.csv'))

class_info = pd.read_csv(os.path.join(DATA_FOLDER,'stage_2_detailed_class_info.csv'))

print(f'Train Labels dataframe has {train_labels.shape[0]} rows and {train_labels.shape[1]} columns')

print(f'Class info dataframe has {class_info.shape[0]} rows and {class_info.shape[1]} columns')

print('Number of duplicates in patientID in train labels dataframe: {}'.format(len(train_labels) - (train_labels['patientId'].nunique())))

print('Number of duplicates in patientID in class info dataframe: {}'.format(len(class_info) - (class_info['patientId'].nunique())))
print('Train labels dataframe:\n'); display(train_labels.head())

print('\nClass info dataframe:\n'); display(class_info.head())
print('Checking value counts for the targets: {}'.format(train_labels['Target'].value_counts().to_dict()))
def fetchDCMFileInfo(patient_id):

    dcm_file = Folder_DCM_IMAGES + '{}.dcm'.format(patient_id)

    dcm_data = dcm.read_file(dcm_file)

    return dcm_data



print(fetchDCMFileInfo(train_labels['patientId'][1]))
patientId = train_labels['patientId'][0]

patient = train_labels.loc[train_labels['patientId'] == patientId].iloc[0]

patient
def showImage(patient):

    path = Folder_DCM_IMAGES + '{}.dcm'.format(patient['patientId'])

    dsimage = dcm.dcmread(path)

    plt.figure(figsize= (5,5))  

    plt.imshow(dsimage.pixel_array, cmap = plt.cm.bone)

    #print(f'patient_id: {patient["patientid"]}')

    plt.show()



patientId = train_labels['patientId'][1]

patient = train_labels.loc[train_labels['patientId'] == patientId].iloc[0]

showImage(patient)
def count_missing_data(data_df):

    total = data_df.isnull().sum().sort_values(ascending = False)

    #percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)

    return pd.concat([total], axis=1, keys=['Total'])
count_missing_data(train_labels)
count_missing_data(class_info)
f, ax = plt.subplots(1,1, figsize=(8,8))

sns.countplot(class_info['class'],order = class_info['class'].value_counts().index)



total = float(len(class_info))

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2,

            height + 113,'{:d}'.format(height),

            ha="center") 

plt.show()
train_class_df = train_labels.merge(class_info, left_on='patientId', right_on='patientId', how='inner')

train_class_df.sample(5)
train_class_df.describe()
train_class_df.info()
train_class_df['class'].value_counts()
train_class_df.isna().sum()
train_class_df.isnull().sum()
train_class_df.Target.value_counts()
#number of patient ids having multiple id's with there count of entries

train_class_df['patientId'].value_counts().value_counts()
# check for duplicate reconrds in training set

print("Unique patientId in  train_class_df: ", train_class_df['patientId'].nunique())
print("No of rows in train_class_df: ", train_class_df.shape[0])
train_class_df.Target.value_counts()
tmp = train_class_df.groupby(['patientId','Target', 'class'])['patientId'].count()

df = pd.DataFrame(data={'Records': tmp.values}, index=tmp.index).reset_index()

dupcount = df.groupby(['Records','Target','class']).count()

dupcount
# getting duplicate count for Target == 1; pneumonia present

train_class_df[train_class_df['Target'] == 1]['patientId'].value_counts().value_counts()
# getting duplicate count for Target == 0; pneumonia absent

train_class_df[train_class_df['Target'] == 0]['patientId'].value_counts().value_counts()
#train_class_df.drop_duplicates(inplace=True)

#train_class_df.reset_index(inplace=True)
train_class_df.info()
# Class info for Pneumonia and Non Pneumonia cases

train_class_df['class'].value_counts()
# Let's plot the number of patients for each class grouped by Target value.

fig, ax = plt.subplots(nrows=1,figsize=(12,6))

tmp = train_class_df.groupby('Target')['class'].value_counts()

df = pd.DataFrame(data={'Counts': tmp.values}, index=tmp.index).reset_index()

sns.barplot(ax=ax,x = 'Target', y='Counts',hue='class',data=df)

plt.title("Class and Target")

plt.show()
# Determining if there are x,y values of bounding boxes info for Pneumonia and Non Pneumonia cases

print ('X value count of bounding box information for pnuemonia Cases: ',train_class_df[train_class_df.Target==1]['x'].count())

print ('X value count of bounding box information for Non-pnuemonia Cases: ',train_class_df[train_class_df.Target==0]['x'].count())
target1 = train_class_df[train_class_df['Target']==1]

plt.figure()

fig, ax = plt.subplots(2,2,figsize=(12,12))

sns.distplot(target1['x'],kde=True,bins=50, color="red", ax=ax[0,0])

sns.distplot(target1['y'],kde=True,bins=50, color="green", ax=ax[0,1])

sns.distplot(target1['width'],kde=True,bins=50, color="blue", ax=ax[1,0])

sns.distplot(target1['height'],kde=True,bins=50, color="yellow", ax=ax[1,1])

locs, labels = plt.xticks()

plt.tick_params(axis='both')

plt.show()
def fetch_image_details(i,data_row,f, ax):

        patientImage = data_row['patientId']+'.dcm'

        imagePath = os.path.join(Folder_DCM_IMAGES,patientImage)

        data_row_img_data = dcm.read_file(imagePath)

        modality = data_row_img_data.Modality

        age = data_row_img_data.PatientAge

        sex = data_row_img_data.PatientSex

        data_row_img = dcm.dcmread(imagePath)

        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}'.format(

                data_row['patientId'],modality, age, sex, data_row['Target'], data_row['class']))



def show_dicom_images(data):

    img_data = list(data.T.to_dict().values())

    f, ax = plt.subplots(2,3, figsize=(16,12))

    for i,data_row in enumerate(img_data):

        fetch_image_details(i,data_row,f, ax)

    plt.show()
print("Images with target = 0 (Not having pneunmonia, class = No lung opacity / Not Normal)")

show_dicom_images(train_class_df[train_class_df['Target']==0].sample(6))
print("Images with target = 1 (having pneunmonia, class = lung opacity)")

show_dicom_images(train_class_df[train_class_df['Target']==1].sample(6))
from matplotlib.patches import Rectangle

import matplotlib.patches as patches

def show_dicom_images_with_boxes(data):

    img_data = list(data.T.to_dict().values())

    f, ax = plt.subplots(2,3, figsize=(16,12))

    for i,data_row in enumerate(img_data):

        fetch_image_details(i,data_row,f, ax)

        rows = train_class_df[train_class_df['patientId']==data_row['patientId']]

        box_data = list(rows.T.to_dict().values())

        for j, row in enumerate(box_data):

            ax[i//3, i%3].add_patch(patches.Rectangle(xy=(row['x'], row['y']),

                        width=row['width'],height=row['height'],  linewidth=2, edgecolor='r', facecolor='none'))

    plt.show()
show_dicom_images_with_boxes(train_class_df[train_class_df['Target']==1].sample(6))
#01b15f07-1149-4ff8-9756-bc821e41b97c.dcm

print(fetchDCMFileInfo('01b15f07-1149-4ff8-9756-bc821e41b97c'))

print(fetchDCMFileInfo('ce84731f-95b7-473e-ad1b-125523a3a71e'))
import os

import csv

import random

import pydicom

import numpy as np

import pandas as pd

from skimage import io

from skimage import measure

from skimage.transform import resize



import tensorflow as tf

from tensorflow import keras



from matplotlib import pyplot as plt

import matplotlib.patches as patches
# empty dictionary

pneumonia_locations = {}

# load table

with open(os.path.join(DATA_FOLDER+'/stage_2_train_labels.csv'), mode='r') as infile:

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
#load and shuffle filenames

folder = Folder_DCM_IMAGES #PATH+'/stage_2_train_images'

filenames = os.listdir(folder)

random.shuffle(filenames)

# split into train and validation filenames

n_valid_samples = 2560

train_filenames = filenames[n_valid_samples:]

valid_filenames = filenames[:n_valid_samples]

print('n train samples', len(train_filenames))

print('n valid samples', len(valid_filenames))

n_train_samples = len(filenames) - n_valid_samples

print('Total train images:',len(filenames))

print('Images with pneumonia:', len(pneumonia_locations))
class generator(keras.utils.Sequence):

    

    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32, image_size=128, shuffle=True, augment=False, predict=False):

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



def create_network(input_size, channels, n_blocks=2, depth=4):

    # input

    inputs = keras.Input(shape=(input_size, input_size, 1))

    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)

    # residual blocks

    for d in range(depth):

        channels = channels * 2

        x = create_downsample(channels, x)

        for b in range(n_blocks):

            x = create_resblock(channels, x)

    # output

    x = keras.layers.BatchNormalization(momentum=0.9)(x)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    outputs = keras.layers.UpSampling2D(2**depth)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

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



def cosine_annealing(x):

    lr = 0.001

    epochs = 25

    return lr*(np.cos(np.pi*x/epochs)+1.)/2

learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)
# from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

# from tensorflow.keras.layers import Concatenate, Conv2D, UpSampling2D, Reshape

# from tensorflow.keras.models import Model



# IMAGE_HEIGHT = 1024

# IMAGE_WIDTH = 1024

# image_size = 224

# ALPHA = 1 # Width hyper parameter for MobileNet (0.25, 0.5, 0.75, 1.0). Higher width means more accurate but slower



# HEIGHT_CELLS = 128

# WIDTH_CELLS = 128



# CELL_WIDTH = IMAGE_WIDTH / WIDTH_CELLS

# CELL_HEIGHT = IMAGE_HEIGHT / HEIGHT_CELLS



# EPOCHS = 1

# BATCH_SIZE = 4

# PATIENCE = 10



# THREADS = 1



# def create_modelUnet(trainable=True):

#      model = MobileNet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), include_top=False, alpha=ALPHA, weights="imagenet")



#     for layer in model.layers:

#         layer.trainable = trainable



#     block1 = model.get_layer("conv_pw_5_relu").output

#     block2 = model.get_layer("conv_pw_11_relu").output

#     block3 = model.get_layer("conv_pw_13_relu").output



#     x = Concatenate()([UpSampling2D()(block3), block2])

#     x = Concatenate()([UpSampling2D()(x), block1])



#     x = Conv2D(1, kernel_size=1, activation="sigmoid")(x)

#     x = Reshape((HEIGHT_CELLS, WIDTH_CELLS))(x)



#     return Model(inputs=model.input, outputs=x)



# def dice_coefficient(y_true, y_pred):

#     numerator = 2 * tf.reduce_sum(y_true * y_pred)

#     denominator = tf.reduce_sum(y_true + y_pred)



#     return numerator / (denominator + tf.keras.backend.epsilon())



# def loss(y_true, y_pred):

#     return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)


# create network and compiler

model = create_network(input_size=128, channels=32, n_blocks=2, depth=4) #create_model(True) # 



                       

# model.compile(optimizer='adam',

#               loss=iou_bce_loss,

#               metrics=['accuracy', mean_iou])





#model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy', mean_iou]) # Regression loss is MSE



#model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy', mean_iou]) # Regression loss is MSE



model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy', mean_iou]) # Regression loss is MSE





#model = create_modelUnet(create_model(True)) # 



print("model summary:", model.summary())

# cosine learning rate annealing
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# model = create_model(False)





# optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# model.compile(loss=loss, optimizer=optimizer, metrics=[dice_coefficient])



# model.summary()

PATIENCE = 10



checkpoint = ModelCheckpoint(SAVE_PATH + "model-{val_loss:.2f}.h5", monitor="val_loss", verbose=1, save_best_only=True,

                             save_weights_only=True, mode="auto", period=1)

stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="auto")

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="auto")
# create train and validation generators

folder = Folder_DCM_IMAGES #PATH+'/stage_2_train_images'

train_gen = generator(folder, train_filenames[0:500], pneumonia_locations, batch_size=16, image_size=128, shuffle=True, augment=True, predict=False)

valid_gen = generator(folder, valid_filenames[0:100], pneumonia_locations, batch_size=16, image_size=128, shuffle=False, predict=False)

history = model.fit_generator(generator=train_gen,

                    epochs=5,

                    validation_data=valid_gen,

                    callbacks=[checkpoint, reduce_lr, stop],

                    workers=4,

                    use_multiprocessing=False,

                    shuffle=True)
import pickle

history_data_pickle_file = SAVE_PATH + "history_data.pickle" 

with open(history_data_pickle_file, "wb") as file_:

    pickle.dump(history.history, file_, -1)
datafrompickle = pickle.load(open(history_data_pickle_file, "rb", -1))

datafrompickle

modelHistory = datafrompickle
plt.figure(figsize=(25,6))

plt.subplot(131)

plt.plot(modelHistory.epoch, modelHistory.history["loss"], label="Train loss")

plt.plot(modelHistory.epoch, modelHistory.history["val_loss"], label="Valid loss")

plt.legend()

plt.subplot(132)

plt.plot(modelHistory.epoch, modelHistory.history["accuracy"], label="Train accuracy")

plt.plot(modelHistory.epoch, modelHistory.history["val_accuracy"], label="Valid accuracy")

plt.legend()

plt.subplot(133)

plt.plot(modelHistory.epoch, modelHistory.history["mean_iou"], label="Train iou")

plt.plot(modelHistory.epoch, modelHistory.history["val_mean_iou"], label="Valid iou")

plt.legend()

plt.show()
# load and shuffle filenames

folder = '/stage_2_test_images'

test_filenames = os.listdir(folder)[:100]

print('n test samples:', len(test_filenames))



# create test generator with predict flag set to True

test_gen = generator(folder, test_filenames, None, batch_size=25, image_size=image_dimension, shuffle=False, predict=True)



# create submission dictionary

submission_dict = {}

# loop through testset

for imgs, filenames in test_gen:

    # predict batch of images

    preds = model.predict(imgs)

    # loop through batch

    for pred, filename in zip(preds, filenames):

        # resize predicted mask

        pred = resize(pred, (1024, 1024), mode='reflect')

        # threshold predicted mask

        comp = pred[:, :, 0] > 0.5

        # apply connected components

        comp = measure.label(comp)

        # apply bounding boxes

        predictionString = ''

        for region in measure.regionprops(comp):

            # retrieve x, y, height and width

            y, x, y2, x2 = region.bbox

            height = y2 - y

            width = x2 - x

            # proxy for confidence score

            conf = np.mean(pred[y:y+height, x:x+width])

            # add to predictionString

            predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '

        # add filename and predictionString to dictionary

        filename = filename.split('.')[0]

        submission_dict[filename] = predictionString

    # stop if we've got them all

    if len(submission_dict) >= len(test_filenames):

        break



# save dictionary as csv file

sub = pd.DataFrame.from_dict(submission_dict,orient='index')

sub.index.names = ['patientId']

sub.columns = ['PredictionString']

sub.to_csv(SAVE_PATH+'pneumonia_model_submission.csv')
from sklearn.metrics import confusion_matrix

pred = model.predict(x_test)

pred = np.argmax(pred,axis = 1) 

y_true = np.argmax(y_test,axis = 1)