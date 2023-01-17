#Importing libraries



import pandas as pd

import numpy as np

import cv2

import tensorflow as tf

import keras

import h5py

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.models import Sequential, Model

from keras.layers import Conv2D, Flatten, MaxPool2D, Input, Dropout, Dense

from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from keras.optimizers import Adam

import os

from pathlib import Path

import glob

from sklearn.model_selection import train_test_split

import cv2

import matplotlib.pyplot as plt

import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



print('Imported Libraries')



print('Reading the data')

#Reading the data

member = ('/kaggle/input/garbage-classification/Garbage classification/Garbage classification/')

catagories = os.listdir(member)

list_items = []

for cat in catagories:

    catagory_img = (member  + cat)

    for _ in (glob.glob(catagory_img +'/'+'*.jpg')):

        list_items.append([cat, _])

    

#Convert list into dataframe

data = pd.DataFrame(list_items,columns = ['catagory', 'filepath'], index = None)

data = data.sample(frac=1).reset_index(drop=True)

data.head(5)

data.shape





#print('Splitting the dataset')

train_data = data[1:2000]

val_data = data[2001:2200]

test_data = data[2201:2527]

#Preprocessing for Training set



def adv_preprocessing(image):

    #loading imageswith



    preimgs = []

    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)



    #Setting dimensions to resize

    height = 224

    width = 224

    

    dim = (width, height)

    res = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)

    preimgs.append(res)

        

#Removing noise from image - Gaussian blur

    

    blurred_img = cv2.GaussianBlur(res, (5,5),0)

    preimgs.append(blurred_img)



    #Segmentation 

    #------------------------------------------------------------------

    image = res

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ret,thresh = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

    

    #More noise removal

    #------------------------------------------------------------------

    kernal = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal, iterations=2)

    

    #Sure background area

    sure_bg = cv2.dilate(opening, kernal, iterations = 3)

    

    #Finding foreground area

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    

    # Finding unknown region

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    

    #Seperating different objects with different backgrounds

    #Markers labelling

    ret, markers  = cv2.connectedComponents(sure_fg)

    #Add one to all labels so that sure background is 0 not 1

    markers = markers+1

    

    #Mark the unknown region with 0

    markers[unknown == 255] = 0

    

    markers = cv2.watershed(res, markers)

    res[markers == -1] = [255,0,0]

    placeholder = np.random.rand(224,224)

    #Displaying the markers on image

    markers = np.dstack([markers,np.zeros((224,224)), placeholder])

    #Adding 

    preimgs.append(res)

    preimgs.append(markers)

    

    return preimgs





#----------------------------------------------------------------------------------------------#

#Preprocessing for Validation and test data



def norm_data(data):

    batch_data = np.zeros((len(data),224,224,3))

    batch_labels = np.zeros((len(data),6))

    label_enc = LabelEncoder()

    y = label_enc.fit_transform(data['catagory']) #Converting data into labels

    y = y.reshape(-1,1)

    onehotenc = OneHotEncoder(handle_unknown= 'ignore')

    batch_labels = onehotenc.fit_transform(y).toarray() #Onehot Encoding data

    

    #Image normalization

    height = 224

    width = 224

    dim = (height, width)

    

    for i in range(len(data)):

        img = cv2.imread(data.iloc[i]['filepath'])

        res_img = cv2.resize(img, dim,interpolation = cv2.INTER_LINEAR)

        res_img = res_img.astype(np.float32)/255

        batch_data[i] = res_img



    return batch_data, batch_labels
#Generator for training data



def adv_gendata(data, batch_size):

    labelenc = LabelEncoder()

    n = len(data)

    steps = n//batch_size

    data['labels'] = labelenc.fit_transform(data['catagory'])

    enc = OneHotEncoder(handle_unknown='ignore')

    enc_df = pd.DataFrame(enc.fit_transform(data[['labels']]).toarray())



    #Defining numpy array to contain label and image data

    

    batch_data = np.zeros((batch_size, 224,224,3), dtype = np.float32)

    batch_labels = np.zeros((batch_size, 6))

    indices = np.arange(n)



    c1 = 0

    i = 0

    #Initialize counter

    while True:

        np.random.shuffle(indices)

        next_batch = indices[(i*batch_size):(i+1)*batch_size]

        count = 0

        



        for j, idx in enumerate(next_batch):

            if count <= batch_size-4:

                img_name = data.iloc[idx]['filepath']

                aug_img = adv_preprocessing(data['filepath'].iloc[idx])

                encoded_label = enc_df.iloc[idx]

                batch_data[count+0] = aug_img[0]

                batch_labels[count+0] = encoded_label

                batch_data[count+1] = aug_img[1]

                batch_labels[count+1] = encoded_label

                batch_data[count+2] = aug_img[2]

                batch_labels[count+2] = encoded_label

                batch_data[count+3] = aug_img[3]

                batch_labels[count+3] = encoded_label

                count +=4

                c1 = c1+1

                

            else:

                count+=1



            if count==batch_size:

                i += 1

                break



        i+=1

        yield batch_data, batch_labels

    

    if i>=steps:

        i=0   



print('Building the model')





#Working on the model

def build_model():

    model = Sequential()

    input_size  = Input(shape = (224,224,3), name  =  'Input_Image')



    #Layer 1 - Deapth Layer 1,2

    x = Conv2D(64,(3,3), activation = 'relu', padding = 'same', name = 'ConvLayer1' )(input_size)

    x = Conv2D(64,(3,3), activation = 'relu', padding = 'same', name = 'ConvLayer2' )(x)

    x = MaxPool2D((2,2), name = 'Maxpool1')(x)

    x = BatchNormalization(name = 'bn1')(x)



    #Layer 2 - Deapth layer 3,4

    x = Conv2D(128,(3,3), activation = 'relu', padding = 'same', name = 'ConvLayer3')(x)

    x = Conv2D(128,(3,3), activation = 'relu', padding = 'same', name = 'ConvLayer4')(x)

    x = MaxPool2D((2,2), name = 'Maxpoo12')(x)

    x = BatchNormalization(name = 'bn2')(x)

#     #Layer 3 - Deapth layer 3

    x = Conv2D(256,(3,3), activation= 'relu',padding = 'same',  name = 'ConvLayer5')(x)

    x = MaxPool2D((2,2), name = 'Maxpool3')(x)

    x = BatchNormalization(name = 'bn3')(x)



    #Flatten the model



    x = Flatten(name = 'Flatten')(x)

    x = Dense(256, activation = 'relu', name = 'FC1')(x)

    x = Dropout(0.7, name = 'Dropout1')(x)

    x = Dense(256, activation = 'relu', name = 'FC2')(x)

    x = Dropout(0.7, name = 'Dropout2')(x)

    x = Dense(6, activation = 'softmax', name = 'Fc3')(x)

    

    model = Model(input = input_size , output = x)

    return model



#Building the model and summary



model = build_model() 

model.summary()



#Initialize the first layer with the weights of Imagenet



f = h5py.File('/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')



#Selecting the layers for which weights needs to be set

w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']

model.layers[1].set_weights = [w,b]



w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']

model.layers[2].set_weights = [w,b]



f.close()

model.summary()   





#Compiling  the model

from keras.optimizers import SGD

opt = SGD(lr=0.01)

model.compile(loss = "categorical_crossentropy", optimizer = opt)

es = EarlyStopping(patience=5)

chkpt = ModelCheckpoint(filepath= 'bestmodel',save_best_only=True, save_weights_only=True)

model.compile(loss= 'binary_crossentropy', metrics= ['accuracy'], optimizer= opt)
#Training the model



batch_size = 20

train_data_gen = adv_gendata(train_data,20)

val_data, val_labels = norm_data(val_data)

test_data, test_labels = norm_data(test_data)



nb_train_steps = train_data.shape[0]//batch_size



epochs = 10

model.fit_generator(train_data_gen,epochs=epochs, verbose = 1, steps_per_epoch=nb_train_steps, validation_data=(val_data, val_labels))



#Evaluating the model on test data



score = model.evaluate(test_data, test_labels, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
