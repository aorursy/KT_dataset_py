# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import random

import pydicom

import cv2



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



import tensorflow as tf

import keras

from keras.models import Model

from keras.utils import to_categorical

from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, GlobalAveragePooling2D

from keras.optimizers import SGD, Adam, RMSprop

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras import backend as K



print(os.listdir('../input/'))

print(os.listdir('../input/rsna-intracranial-hemorrhage-detection/'))
# Constants

SEED = 42

NUM_CLASSES = 6

NO_PATIENTS_SELECTED = 10000 #50000

IMG_DIM = 512

NO_CHANNEL = 3

BATCH_SIZE = 64



PATH='../input/rsna-intracranial-hemorrhage-detection/'



RESNET_WEIGHT_FULLPATH = '../input/resnet50weightsfile/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

TRAIN_CSV_FULLPATH = PATH + 'stage_1_train.csv'

TEST_CSV_FULLPATH = PATH + 'stage_1_sample_submission.csv'

TRAIN_IMG_PATH = PATH + 'stage_1_train_images/'

TEST_IMG_PATH = PATH + 'stage_1_test_images/'





random.seed(SEED) 
df_train = pd.read_csv(TRAIN_CSV_FULLPATH)



# Removing corrupted image

df_train = df_train[df_train.ID != 'ID_6431af929_epidural']

df_train = df_train[df_train.ID != 'ID_6431af929_intraparenchymal']

df_train = df_train[df_train.ID != 'ID_6431af929_intraventricular']

df_train = df_train[df_train.ID != 'ID_6431af929_subarachnoid']

df_train = df_train[df_train.ID != 'ID_6431af929_subdural']

df_train = df_train[df_train.ID != 'ID_6431af929_any']
df_train.head(50)
df_test = pd.read_csv(TEST_CSV_FULLPATH)
df_test.head(10)
# Randomly sample 50,000 the patientIDs



total_train_patient_no = int(df_train.shape[0]/NUM_CLASSES)

print(total_train_patient_no)



total_test_patient_no = int(df_test.shape[0]/NUM_CLASSES)

print(total_test_patient_no)



train_sample_patient_idx_list = random.sample(range(0,total_train_patient_no), NO_PATIENTS_SELECTED)

print(len(train_sample_patient_idx_list))
# Create new train Dataframe for training resnet with multilabel data



df_train_multilbl = pd.DataFrame(

    columns=['ID','patientID','epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'])



cnt = 1

for patient_idx in train_sample_patient_idx_list:

    k = patient_idx*6

    ID_col_val =  df_train.iloc[k]['ID']

    img_name = 'ID_' + ID_col_val.split('_')[1] + '.dcm'

    patientID = ID_col_val.split('_')[1]

    if len(ID_col_val.split('_')) == 3:

        print(cnt)

        cnt += 1

        epidural_lbl = df_train.iloc[k]['Label']

        intraparenchymal_lbl = df_train.iloc[k+1]['Label']

        intraventricular_lbl = df_train.iloc[k+2]['Label']

        subarachnoid_lbl = df_train.iloc[k+3]['Label']

        subdural_lbl = df_train.iloc[k+4]['Label']

        any_lbl = df_train.iloc[k+5]['Label']

        df_train_multilbl = df_train_multilbl.append(

            {'ID': img_name, 

             'patientID': patientID,

             'epidural': epidural_lbl, 

             'intraparenchymal': intraparenchymal_lbl, 

             'intraventricular': intraventricular_lbl, 

             'subarachnoid': subarachnoid_lbl, 

             'subdural': subdural_lbl, 'any': any_lbl}, ignore_index=True)



print(df_train_multilbl.shape)

df_train_multilbl.head(10)
print(df_train_multilbl.shape)

df_train_multilbl.head(10)



print(any(df_train_multilbl['epidural']==1))

print(any(df_train_multilbl['intraparenchymal']==1))

print(any(df_train_multilbl['intraventricular']==1))

print(any(df_train_multilbl['subarachnoid']==1))

print(any(df_train_multilbl['subdural']==1))

print(any(df_train_multilbl['any']==1))

# Create new test Dataframe for testing resnet with multilabel data

'''

df_test_multilbl = pd.DataFrame(

    columns=['ID','patientID','epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'])



cnt = 1

for k in range(0, total_test_patient_no, NUM_CLASSES):

    ID_col_val =  df_train.iloc[k]['ID']

    img_name = 'ID_' + ID_col_val.split('_')[1] + '.dcm'

    patientID = ID_col_val.split('_')[1]

    if len(ID_col_val.split('_')) == 3:

        print(cnt)

        cnt += 1

        epidural_lbl = df_test.iloc[k]['Label']

        intraparenchymal_lbl = df_test.iloc[k+1]['Label']

        intraventricular_lbl = df_test.iloc[k+2]['Label']

        subarachnoid_lbl = df_test.iloc[k+3]['Label']

        subdural_lbl = df_test.iloc[k+4]['Label']

        any_lbl = df_test.iloc[k+5]['Label']

        df_test_multilbl.append(

            {'ID': img_name, 

             'patientID': patientID,

             'epidural': epidural_lbl, 

             'intraparenchymal': intraparenchymal_lbl, 

             'intraventricular': intraventricular_lbl, 

             'subarachnoid': subarachnoid_lbl, 

             'subdural': subdural_lbl, 'any': any_lbl}, ignore_index=True)

        

print(df_test_multilbl.shape())

df_test_multilbl.head(10)

'''
# Network Architecture



input_tensor = Input(shape=(224, 224, NO_CHANNEL))

        

base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

base_model.load_weights(RESNET_WEIGHT_FULLPATH)



    # add a global spatial average pooling layer

x = base_model.output

output = BatchNormalization()(x)

x = GlobalAveragePooling2D()(x)



# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)



# what should be the last layer activation function for multilabel learning with 6 classes?

predictions = Dense(NUM_CLASSES, activation='softmax')(x) 



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional Resnet50 layers

for layer in base_model.layers:

    layer.trainable=False

    

for layer in model.layers:

    layer.trainable=True

    

print(model.summary())
# split train data into train and dev set



y_cols = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']

all_cols = df_train_multilbl.columns

x_cols = list(set(all_cols).difference(set(y_cols)))



X = df_train_multilbl[x_cols]

y = df_train_multilbl[y_cols]



x_train, x_val, y_train, y_val = train_test_split(X, y,test_size=0.33, random_state=SEED)



del df_train, df_test, df_train_multilbl, X, y

gc.collect()



print(x_train.shape)

print(y_train.shape)



print(x_val.shape)

print(y_val.shape)
# pre-processing function for the image generators



def window_img(dcm, width=None, level=None):

    pixels = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    if not width:

        width = dcm.WindowWidth

        if type(width) != pydicom.valuerep.DSfloat:

            width = width[0]

    if not level:

        level = dcm.WindowCenter

        if type(level) != pydicom.valuerep.DSfloat:

            level = level[0]

    lower = level - (width / 2)

    upper = level + (width / 2)

    return np.clip(pixels, lower, upper)



def load_one_image(img_fullpath, width=200, level=80):

    dcm_data = pydicom.dcmread(img_fullpath)

    #assert('filepath' in dcm_data.columns)

    #pixels = window_img(dcm_data, width, level)

    #print(dcm_data.pixel_array.shape)

    return dcm_data.pixel_array



def load_and_normalize_dicom(path, x, y, n):

    dicom1 = pydicom.read_file(path)

    dicom_img = dicom1.pixel_array.astype(np.float64)

    mn = dicom_img.min()

    mx = dicom_img.max()

    if (mx - mn) != 0:

        dicom_img = (dicom_img - mn)/(mx - mn)

    else:

        dicom_img[:, :] = 0

    if dicom_img.shape != (x, y):

        dicom_img = cv2.resize(dicom_img, (x, y), interpolation=cv2.INTER_CUBIC)

    

    if n == 3:

        image = np.stack([dicom_img,dicom_img,dicom_img])

        image = image.reshape(x, y, n)

        

    return image



sample_img_arr = ['ID_63eb1e259.dcm', 'ID_2669954a7.dcm', 'ID_52c9913b1.dcm', 'ID_4e6ff6126.dcm']

img_fullpath = TRAIN_IMG_PATH + sample_img_arr[2]



pixels = load_one_image(img_fullpath)

plt.imshow(pixels)
# Define customized data generator for training and validation



#y_cols = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']



def batch_generator_train_multilbl(path, df, x_col, y_cols, batch_size, target_size):

    number_of_batches = np.ceil((df.shape[0])/batch_size)

    counter = 0

    #random.shuffle(df)

    df_idx_lst = range(0,df.shape[0])

    no_batches = df.shape[0]//batch_size

    while True:

        df_idx_batch = df_idx_lst[batch_size*counter:batch_size*(counter+1)]

        image_list = []

        mask_list = []

        for idx in df_idx_batch:

            filename = df.iloc[idx][x_col]

            image = load_and_normalize_dicom(path + filename, target_size[0], target_size[1], NO_CHANNEL)

            mask = []

            for col in y_cols:

                is_subtype = int(df.iloc[idx][col])

                mask.append(is_subtype)

            image = np.stack([image,image,image])

            image = image.reshape((target_size[0], target_size[1], NO_CHANNEL))

            image_list.append(image)

            mask_list.append(mask)

        counter += 1

        image_list = np.array(image_list)

        mask_list = np.array(mask_list)

        #print(image_list.shape)

        #print(mask_list.shape)

        yield (image_list, mask_list)

        del image_list, mask_list

        gc.collect()

        if counter == no_batches-1:

            counter = 0
# Code for creating training and validation data generator



'''train_datagen = image.ImageDataGenerator(rescale=1. / 255, validation_split=0.15, horizontal_flip=True,

                                         vertical_flip=True, rotation_range=360, zoom_range=0.2, shear_range=0.1#,

                                         #preprocessing_function=load_one_image 

                                        )



train_generator = train_datagen.flow_from_dataframe(dataframe = pd.concat([x_train, y_train], axis=1),

                                                    directory = TRAIN_IMG_PATH,

                                                    x_col = 'ID',

                                                    y_col = y_cols,

                                                    batch_size = BATCH_SIZE,

                                                    class_mode = 'multi_output',

                                                    target_size = (IMG_DIM, IMG_DIM),

                                                    subset = 'training',

                                                    shuffle = True,

                                                    seed = SEED,

                                                    validate_filenames = False

                                                    )



valid_generator = train_datagen.flow_from_dataframe(dataframe = pd.concat([x_val, y_val], axis=1),

                                                    directory = TRAIN_IMG_PATH,

                                                    x_col = 'ID',

                                                    y_col = y_cols,

                                                    batch_size = BATCH_SIZE,

                                                    class_mode = 'multi_output',

                                                    target_size = (IMG_DIM, IMG_DIM),

                                                    subset = 'validation',

                                                    shuffle = True,

                                                    seed = SEED,

                                                    validate_filenames = False

                                                    )

'''



# Code for creating testing data generator

# Configuration for the Network



conf = dict()

# Change this variable to 0 in case you want to use full dataset

conf['use_sample_only'] = 1

# Save weights

conf['save_weights'] = 0

# How many patients will be in train and validation set during training. Range: (0; 1)

conf['train_valid_fraction'] = 0.5

# Batch size for CNN [Depends on GPU and memory available]

conf['batch_size'] = 200

# Number of epochs for CNN training

conf['nb_epoch'] = 40

# Early stopping. Stop training after epochs without improving on validation

conf['patience'] = 3

# Shape of image for CNN (Larger the better, but you need to increase CNN as well)

conf['image_shape'] = (64, 64)

# Learning rate for CNN. Lower better accuracy, larger runtime.

conf['learning_rate'] = 1e-2

# Number of random samples to use during training per epoch 

conf['samples_train_per_epoch'] = 10000

# Number of random samples to use during validation per epoch

conf['samples_valid_per_epoch'] = 1000

# Some variables to control CNN structure

conf['level_1_filters'] = 4

conf['level_2_filters'] = 8

conf['dense_layer_size'] = 32

conf['dropout_value'] = 0.5
# Alternative Data generator

from tensorflow.python.keras.utils import Sequence

import numpy as np  



class image_data_generator_multilbl(Sequence):

    def __init__(self, x_set, y_set, batch_size):

        self.x, self.y = x_set, y_set

        self.batch_size = batch_size



    def __len__(self):

        return int(np.ceil(len(self.x) / float(self.batch_size)))



    def __getitem__(self, idx):

        x_col = 'ID'

        y_cols = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']

        path = TRAIN_IMG_PATH

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        #print(batch_y.shape)

        # read your data here using the batch lists, batch_x and batch_y

        x = [load_and_normalize_dicom(path+filename,224,224,NO_CHANNEL) for filename in batch_x[x_col]] 

        y = []

        for k in range(0,self.batch_size):

            subtypes_lst = []

            for col in y_cols:

                #print(batch_y.iloc[k][col])

                is_subtype = int(batch_y.iloc[k][col])

                subtypes_lst.append(is_subtype) 

            y.append(np.array(subtypes_lst))

        return (np.array(x), np.array(y))
# Code for defining callbacks: EarlyStopping and ReduceLROnPlateau



eraly_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')

# Reducing the Learning Rate if result is not improving. 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',

                              verbose=1)
# Code for compiling the model

epochs = 10

lrate = 0.01

decay = lrate/epochs

#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(optimizer = Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
'''xy_train = pd.concat([x_train,y_train], axis=1)

xy_val = pd.concat([x_val,y_val], axis=1)

history = model.fit_generator(generator=batch_generator_train_multilbl(path=TRAIN_IMG_PATH, df=xy_train, x_col='ID', y_cols=y_cols, batch_size=BATCH_SIZE, target_size=(224, 224)), 

                              validation_data=batch_generator_train_multilbl(path=TRAIN_IMG_PATH, df=xy_val, x_col='ID', y_cols=y_cols, batch_size=BATCH_SIZE, target_size=(224, 224)), 

                              epochs = epochs, 

                              steps_per_epoch=xy_train.shape[0],#//BATCH_SIZE,

                              validation_steps=xy_val.shape[0],#//BATCH_SIZE,

                              #samples_per_epoch=BATCH_SIZE,

                              callbacks=[eraly_stop, reduce_lr],

                              use_multiprocessing=True, 

                              workers=4,

                              shuffle=True,

                              verbose=1

                            )'''
#xy_train = pd.concat([x_train,y_train], axis=1)

#xy_val = pd.concat([x_val,y_val], axis=1)

history = model.fit_generator(generator=image_data_generator_multilbl(x_train, y_train, BATCH_SIZE), 

                              validation_data=image_data_generator_multilbl(x_train, y_train, BATCH_SIZE), 

                              epochs = epochs, 

                              #steps_per_epoch=x_train.shape[0]//BATCH_SIZE,

                              #validation_steps=x_val.shape[0]//BATCH_SIZE,

                              callbacks=[eraly_stop, reduce_lr],

                              use_multiprocessing=False, 

                              #workers=4,

                              verbose=1

                            )
L=[[1,2,3],[4,5,6]]

A=np.array([[1,2,3],[4,5,6]])

print(A.shape)

B=np.stack([A,A])

B=B.reshape((2,3,2))

print(B.shape)
train_datagen=batch_generator_train_multilbl(path=TRAIN_IMG_PATH, df=xy_train, x_col='ID', y_cols=y_cols, batch_size=BATCH_SIZE, target_size=(224, 224))

train_datagen
y_train.shape
(y_train[0:64]).shape