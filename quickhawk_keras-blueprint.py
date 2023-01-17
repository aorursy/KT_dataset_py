import pandas

import os

import numpy

from tqdm import tqdm_notebook

import gc

from PIL import Image
!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar
shape = (100, 100, 3)

MAX_VALUE = 140
# Directories :-

PATH = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/'

TRAIN_PATHS = [i for i in os.listdir(PATH) if 'train' in i]

TEST_PATHS = [i for i in os.listdir(PATH) if 'test' in i]



TRAIN_PATHS, TEST_PATHS
!pip install lyft-dataset-sdk

from pyquaternion import Quaternion

from lyft_dataset_sdk.lyftdataset import LyftDataset

lyft_data = LyftDataset(data_path = '.', json_path = PATH + 'train_data', verbose = True)
train_data = pandas.read_csv(PATH + 'train.csv')
categories = [i['name'] for i in lyft_data.category]

categories
columns = ['confidence' ,'center_x', 'center_y', "center_z", 'width', 'length', 'height', 'rotate_w', 'rotate_x', 'rotate_y', 'rotate_z', 'class']

sensors = lyft_data.sensor

sensors = [i['channel'] for i in sensors]

sensors = [i for i in sensors if 'LIDAR' not in i]
def getImageFileNames(token : str):

    

    list_of_filenames = []

    

    for sensor in sensors:

        filename = lyft_data.get('sample_data', lyft_data.get('sample', token)['data'][sensor])['filename']

        list_of_filenames.append(filename)

        

    return list_of_filenames        
def getData(token):

    

    list_of_values = []

    list_of_anns = lyft_data.get('sample', token)['anns']

    

    for annotation_token in list_of_anns:

        values = [1.0]

        sample_data = lyft_data.get('sample_annotation', annotation_token)

        values = values + sample_data['translation'] + sample_data['size'] + sample_data['rotation'] + one_hot_encoding(sample_data['category_name'])

        list_of_values.append(values)

        

    for _ in range(MAX_VALUE - len(list_of_anns)):

        list_of_values.append([0]*11 + one_hot_encoding('other_vehicle'))

    

    return numpy.array(list_of_values)
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, concatenate, GlobalAveragePooling2D, add

from tensorflow.keras import Input

from tensorflow.keras.models import Model
from keras import backend as K

import tensorflow as tf



def mse(y_true, y_pred):

     return K.mean(K.square(y_pred - y_true), axis=-1)



def LossFunction(y_true, y_pred):

     return mse(y_true[:,0], y_pred[:,0]) + mse(y_true[:,1:4], y_pred[:,:4]) + mse(y_true[:,4:7], y_pred[:,4:7]) + mse(y_true[:,7:11], y_pred[:,7:11]) + K.categorical_crossentropy(y_true[:,11:], y_pred[:,11:])



def IOU_Metric(true, pred): #any shape can go - can't be a loss function

    

    def iou_loss_core(true,pred):  #this can be used as a loss if you make it negative

        intersection = true * pred

        notTrue = 1 - true

        union = true + (notTrue * pred)



        return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

    

    def castF(x):

        return K.cast(x, K.floatx())



    def castB(x):

        return K.cast(x, bool)

    

    tresholds = [0.5 + (i*.05)  for i in range(10)]



    #flattened images (batch, pixels)

    true = K.batch_flatten(true)

    pred = K.batch_flatten(pred)

    pred = castF(K.greater(pred, 0.5))



    #total white pixels - (batch,)

    trueSum = K.sum(true, axis=-1)

    predSum = K.sum(pred, axis=-1)



    #has mask or not per image - (batch,)

    true1 = castF(K.greater(trueSum, 1))    

    pred1 = castF(K.greater(predSum, 1))



    #to get images that have mask in both true and pred

    truePositiveMask = castB(true1 * pred1)



    #separating only the possible true positives to check iou

    testTrue = tf.boolean_mask(true, truePositiveMask)

    testPred = tf.boolean_mask(pred, truePositiveMask)



    #getting iou and threshold comparisons

    iou = iou_loss_core(testTrue,testPred) 

    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]



    #mean of thressholds for true positives and total sum

    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)

    truePositives = K.sum(truePositives)



    #to get images that don't have mask in both true and pred

    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1

    trueNegatives = K.sum(trueNegatives) 



    return (truePositives + trueNegatives) / castF(K.shape(true)[0])
# Create your own Model

# This is a test model

def custom_model():

    

    inputs = []

    x = []

    for i in range(len(sensors)):

        inputs.append(Input(shape, batch_size = 1, name = 'CAM_INPUT_' + str(i)))



    for i in range(len(sensors)):

        y = Conv2D(64, (3, 3))(inputs[i])

        y = Conv2D(64, (3, 3))(y)

        y = Dropout(0.05)(y)



        y = Conv2D(128, (3, 3))(y)

        y = MaxPool2D()(y)



        x.append(y)

        

    x = add(x)

    x = GlobalAveragePooling2D()(x)



    outputs = []

    for i in range(MAX_VALUE):



        confidence = Dense(1, activation = 'sigmoid', name = 'CONFIDENCE_' + str(i+1))(x)

        center = Dense(3, activation = 'linear', name = 'CENTER_' + str(i+1))(x)

        size = Dense(3, activation = 'linear', name = 'SIZE_' + str(i+1))(x)

        rotation = Dense(4, activation = 'linear', name = 'ROTATION_' + str(i+1))(x)

        category = Dense(10, activation = 'softmax', name = 'CATEGORY_' + str(i+1))(x)



        output_layer = concatenate([confidence, center, size, rotation, category], name = 'OUTPUT_LAYER_' + str(i))

        

        outputs.append(output_layer)

        

    model = Model(inputs, outputs)

    model.compile(loss = LossFunction, optimizer = 'adam', metrics = ['accuracy', IOU_Metric])

    

    return model
def one_hot_encoding(value):

    global categories

    

    x = categories.index(value)

    

    return [0] * (x) + [1] + [0] * (len(categories) - x)
def model_fit(model):

    

    for token in train_data['Id']:

        values = getData(token)

        values = values.reshape((values.shape[0],) + (1, ) + (values.shape[1], ))

        filenames = getImageFileNames(token)

        

        images = [numpy.asarray(Image.open(i).resize(shape[:-1])).reshape((1, ) + shape) for i in filenames]

        

        print(images[0].shape, values.shape)

        print("Training the model . . . . . ")

        model.fit(images, values.tolist())

        print("Model Trained for token : {}".format(token))

        

        del images, values

        gc.collect()
%%time

model = custom_model()

# print(model.summary())
model_fit(model)
# !rm images

# !rm maps

# !rm lidar
# !ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/test_images images

# !ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/test_maps maps

# !ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/test_lidar lidar
# test_lyft_data = LyftDataset(data_path = '.', json_path = PATH + 'test_data', verbose = True)

# test_data = pandas.read_csv(PATH + 'sample_submission.csv')



# x = test_lyft_data.get('sample_data', test_lyft_data.get('sample', test_data['Id'][0])['data']['CAM_BACK'])

# y = test_lyft_data.get('calibrated_sensor', x['calibrated_sensor_token'])

# z = test_lyft_data.get('ego_pose', x['ego_pose_token'])



# y, z, test_lyft_data.get_sample_data(test_lyft_data.get('sample_data', test_lyft_data.get('sample', test_data['Id'][0])['data']['CAM_BACK'])['token'])