# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/oct2017/OCT2017 /test"))



# Any results you write to the current directory are saved as output.
TRAIN_PATH='../input/oct2017/OCT2017 /train'

TEST_PATH='../input/oct2017/OCT2017 /test'

VAL_PATH='../input/oct2017/OCT2017 /val'
import os



import gc

import re

import operator 



import numpy as np

import pandas as pd



from gensim.models import KeyedVectors



from sklearn import model_selection





import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import BatchNormalization

from keras.layers import Dropout





from sklearn.model_selection import train_test_split, cross_val_score

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer





from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, Masking

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint,  Callback, EarlyStopping, ReduceLROnPlateau

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.parallel

import torch.optim as optim

import torch.utils.data

from torch.autograd import Variable
def create_model(in_shape,pool_size,kernal_size):

    inputs=Input(shape=in_shape)

    x=Convolution2D(filters=32,kernel_size=kernal_size,activation='relu')(inputs)

    x=MaxPooling2D(pool_size=pool_size)(x)

    x=Dropout(0.3)(x)

    x=Convolution2D(filters=64,kernel_size=kernal_size,activation='relu')(x)

    x=MaxPooling2D(pool_size=pool_size)(x)

    x=Dropout(0.3)(x)

    x=Convolution2D(filters=128,kernel_size=kernal_size,activation='relu')(x)

    x=MaxPooling2D(pool_size=pool_size)(x)

    x=Dropout(0.3)(x)

    x=Convolution2D(filters=128,kernel_size=kernal_size,activation='relu')(x)

    x=MaxPooling2D(pool_size=pool_size)(x)

    x=Dropout(0.3)(x)

    x=Flatten()(x)

    x=Dense(4,activation='softmax')(x)

    return x,inputs

    
out,ins=create_model(in_shape=(256,256,3),pool_size=(2,2),kernal_size=(3,3))
model=Model(input=ins,output=out)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from keras.callbacks import EarlyStopping

early_stop=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

import warnings

warnings.filterwarnings("ignore")
from keras.preprocessing.image import ImageDataGenerator

def model_trainer(model):

    train_datagen = ImageDataGenerator(rescale = 1./150, 

                                   shear_range = 0.01, 

                                   zoom_range =[0.9, 1.25],

                                   rotation_range=20,

                                   zca_whitening=True,

                                   vertical_flip=True,

                                   fill_mode='nearest',

                                   width_shift_range=0.1,

                                   height_shift_range=0.1,

                                   brightness_range=[0.5, 1.5],

                                   horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./160)



    train_generator = train_datagen.flow_from_directory(

        TRAIN_PATH,

        target_size=(256,256),

        batch_size=32,

        class_mode='categorical')



    validation_generator = test_datagen.flow_from_directory(

        TEST_PATH,

        target_size=(256,256),

        batch_size=32,

        class_mode='categorical')# multiclass then  categorical



    hist=model.fit_generator(

        train_generator,

        steps_per_epoch=2000, # no of images in training set

        epochs=10,

        shuffle=True,

        validation_data=validation_generator,

        validation_steps=968,callbacks=[early_stop]) # no of  images in test

    return hist,train_generator
hist,train_generator=model_trainer(model)

print(hist)

#VISUAL ANALYSIS 

import numpy as np   

from keras.preprocessing import image    

im1_path="../input/oct2017/OCT2017 /test/NORMAL/NORMAL-1025847-1.jpeg"

test_image=image.load_img(im1_path,target_size=(256,256))
import matplotlib.pyplot as plt

plt.imshow(test_image)

# now to convert to 3 dimensional from 2d

test_image=image.img_to_array(test_image)

print(test_image.size)

#test_image=image.img_to_array(test_image)

test_image= np.expand_dims(test_image,axis=0)

train_generator.class_indices
result=np.argmax(model.predict(test_image))

print(result)
def get_name_layer_filters(model):

    filter_whole=[]

    layer_whole=[]

    for layer in model.layers:

        if 'conv' not in layer.name:

            continue

        filters,biases=layer.get_weights()

        filter_whole.append(filters)

        layer_whole.append(biases)

        print(layer.name,filters.shape)

    return filter_whole,layer_whole    

        
filter_whole,layer_whole=get_name_layer_filters(model)
filters,biases=model.layers[1].get_weights()
f_min,f_max=filters.min(),filters.max()

filters=(filters-f_min)/(f_max-f_min)
from matplotlib import pyplot

n_filters,ix=6,1

for i in range(n_filters):

    f=filters[:,:,:,i]

    #Plot each channel

    for j in range(3):

        ax=pyplot.subplot(n_filters,3,ix)

        ax.set_xticks([])

        ax.set_yticks([])

        #Plot filter channel

        pyplot.imshow(f[:,:,j])

        ix+=1

        

pyplot.show()    
#FEATURE MAP

model_feature=Model(inputs=model.inputs,outputs=model.layers[4].output)
#We use result from previous

feature_map=model_feature.predict(test_image)

feature_map.shape
#plot all 32 maps in an 8*4 squares

pyplot.figure(figsize=(30,30))        

        

square=8

ix=1

for _ in range(4):

    for _ in range(8):

        ax=pyplot.subplot(square,square,ix)

        ax.set_xticks([])

        ax.set_yticks([])

        pyplot.imshow(feature_map[0,:,:,ix-1])

        ix+=1

        

pyplot.show()
def get_convolutional_layers(model):

    convolutions_models=[]

    for layer in model.layers:

        if 'conv2d' not in layer.name:

            continue

        model_temp=Model(inputs=model.inputs,outputs=layer.output)

        convolutions_models.append(model_temp)

    return convolutions_models    

        

        
#To see each feature map systematically for every convolutional layer

def generate_feature_maps(model,test_image):

    models=get_convolutional_layers(model)#Fetching convolution layers models

    feature_maps=[]

    

    for model_temp in models:

        feature_map=model_feature.predict(test_image)

        feature_maps.append(feature_map)

    return feature_maps,models    

def plot_graph(feature_map):

    

    #plot all 32 maps in an 8*4 squares

    pyplot.figure(figsize=(30,30))        

        

    square=8

    ix=1

    for _ in range(4):

        for _ in range(8):

            ax=pyplot.subplot(square,square,ix)

            ax.set_xticks([])

            ax.set_yticks([])

            pyplot.imshow(feature_map[0,:,:,ix-1])

            ix+=1

        

    pyplot.show()
def plots_generator(model):

    print("IMAGE UNDER CONSIDERATION")

    test_image=image.load_img(im1_path,target_size=(256,256))

    plt.imshow(test_image)

    test_image=image.img_to_array(test_image)



    test_image= np.expand_dims(test_image,axis=0)

    print()

    feature_maps,models=generate_feature_maps(model,test_image)

    #ax=pyplot.subplot(square,square,ix)# only 32 filters will be shown of each layer

    counter=1

    for each_map in feature_maps:

        print("Convolutional Layer Number {} ".format(counter))

        counter+=1

        #ax=pyplot.subplot(square,square,ix)

        plot_graph(each_map)



plots_generator(model)
hist.history['val_acc']
def val_acc_plot():

    print("FOR MODEL VALIDATION DATA")

    plt.plot(hist.epoch,hist.history['val_acc'])

    plt.xlabel("EPOCHS")

    plt.ylabel("Validation Accuracy")

    
def generate_images(all_paths):

    test_images=[]

    interpret=train_generator.class_indices

    test_y=[]

    

    for path in all_paths:

        y=''

        if 'DME' in path:

            y='DME'

        elif 'DRUSEN' in path:   

            y='DRUSEN'

        elif 'CNV' in path:   

            y='CNV'

        elif 'NORMAL' in path:   

            y='NORMAL'    

        

        for image_path in os.listdir(path):

            new_path=os.path.join('../input/oct2017/OCT2017 /test',y)

            #print(new_path)

            

            new_path=os.path.join(new_path,image_path)

            if 'Store' in str(new_path):

                continue

            temp_images=image.load_img(new_path,target_size=(256,256))

            temp_images=image.img_to_array(temp_images)

            test_images.append(temp_images)

            test_y.append(interpret[y])

    return test_images,test_y        

def generate_predictions(test_images,model):

    predictions=np.argmax(model.predict(test_images),axis=1)

    return predictions

    
from sklearn import metrics



def convert_data(model):

    # Now testing all test images to find test set accuracy.

    #We first need to put all test images and convert them in desired format to predict

    #Firstly we store path of test directory

    PATH_TEST="../input/oct2017/OCT2017 /test"

    all_paths=[]

    print("GENERATING PATHS")

    for directory in os.listdir(PATH_TEST):

        if 'Store' in directory:

            continue

        all_paths.append(os.path.join(PATH_TEST,directory))

    

    test_images,test_y=generate_images(all_paths)

    print("PATH GENERATED")

    test_images=np.array(test_images)

    print("GENERATING PREDICTIONS")

    predictions=generate_predictions(test_images,model)

    print("PREDICTIONS GENERATED")

    print()

    print("ACCURACY OF MODEL FOR TEST DATA IS {}".format(metrics.accuracy_score(test_y, predictions)))

    

    

convert_data(model)