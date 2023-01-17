from sklearn.utils import shuffle

import pandas as pd

import os

# Save train labels to dataframe

df = pd.read_csv("../input/train_labels.csv")



# Save test labels to dataframe

df_test = pd.read_csv('../input/sample_submission.csv')



df = shuffle(df)
# For demonstration only

df = df[:10000]
# Split data set  to train and validation sets

from sklearn.model_selection import train_test_split



# Use stratify= df['label'] to get balance ratio 1/1 in train and validation sets

df_train, df_val = train_test_split(df, test_size=0.1, stratify= df['label'])



# Check balancing

print("Train data: " + str(len(df_train[df_train["label"] == 1]) + len(df_train[df_train["label"] == 0])))

print("True positive in train data: " +  str(len(df_train[df_train["label"] == 1])))

print("True negative in train data: " +  str(len(df_train[df_train["label"] == 0])))

print("Valid data: " + str(len(df_val[df_val["label"] == 1]) + len(df_val[df_val["label"] == 0])))

print("True positive in validation data: " +  str(len(df_val[df_val["label"] == 1])))

print("True negative in validation data: " +  str(len(df_val[df_val["label"] == 0])))
# Train List

train_list = df_train['id'].tolist()

train_list = ['../input/train/'+ name + ".tif" for name in train_list]



# Validation List

val_list = df_val['id'].tolist()

val_list = ['../input/train/'+ name + ".tif" for name in val_list]



# Test list

test_list = df_test['id'].tolist()

test_list = ['../input/test/'+ name + ".tif" for name in test_list]



# Names library

id_label_map = {k:v for k,v in zip(df.id.values, df.label.values)}
# Functions for generators

def get_id_from_file_path(file_path):

    return file_path.split(os.path.sep)[-1].replace('.tif', '')



def chunker(seq, size):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
# Augmentation

from imgaug import augmenters as iaa

import imgaug as ia

import numpy as np 

import cv2



def augmentation():

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(

        [

            # apply the following augmenters to most images

            iaa.Fliplr(0.5), # horizontally flip 50% of all images

            iaa.Flipud(0.5), # vertically flip 20% of all images

            

            # crop images by -10% to 10% of their height/width

            sometimes(iaa.CropAndPad(

                percent=(-0.2, 0.2),

                pad_mode=ia.ALL,

                pad_cval=(0, 255))),

            

            sometimes(iaa.Affine(

                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis

                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)

                rotate=(-45, 45), # rotate by -45 to +45 degrees

                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                cval=(0, 255), # if mode is constant, use a cval between 0 and 255

                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

            )),

            # execute 0 to 5 of the following (less important) augmenters per image

            # don't execute all of them, as that would often be way too strong

            

            iaa.SomeOf((0, 5),

                [    

                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images

                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

                    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)), # add gaussian noise to images

                    iaa.AddToHueAndSaturation((-1, 1)) # change hue and saturation

                ],

                random_order=True

            )

        ],

        random_order=True

    )

    return seq
# Import Pretrained Models

import keras

from keras.applications.densenet import DenseNet201, preprocess_input

from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Concatenate

from keras.models import Model



inputTensor = Input((96,96,3))

model_DenseNet201 = keras.applications.DenseNet201(include_top=False,input_shape = (96,96,3), weights='imagenet')
# Concatenate Pretrained Models

######

#models = [model_DenseNet201, model_ResNet50]

models = [model_DenseNet201]

######



outputTensors = [m(inputTensor) for m in models]

if len(models) > 1:

    output = Concatenate()(outputTensors) 

else:

    output = outputTensors[0]
# Classifier

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.layers import Activation

from keras.models import Sequential

from keras import applications

from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Concatenate



out1 = GlobalAveragePooling2D()(output)

out2 = GlobalMaxPooling2D()(output)

out = Concatenate(axis=-1)([out1, out2])

out = BatchNormalization()(out)

out = Dropout(0.6)(out)

out = Dense(1, activation="sigmoid")(out)

model = Model(inputTensor,out)

model.summary()
# Read, convert and resize iamges

import cv2 as cv

import os



def read_image(path):

    assert os.path.isfile(path), "File not found"

    im = cv.imread(path)

    # Convert from cv2 standard of BGR to our convention of RGB.

    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

    return im
# Train Generator

def train_gen(list_files, id_label_map, batch_size, augment = False):  

    while True:

        

        shuffle(list_files)

        

        for batch in chunker(list_files, batch_size):

            

            X = [read_image(x) for x in batch]

            Y = [id_label_map[get_id_from_file_path(x)] for x in batch]      

            

            if augment:

                X = augmentation().augment_images(X)   

            

            X = [preprocess_input(x) for x in X]  

            yield np.array(X), np.array(Y)



# Validation Generator 



def val_gen(list_files, batch_size):

    

    while True:

        for batch in chunker(list_files, batch_size):

                

            X = [read_image(x) for x in batch]

            X = [preprocess_input(x) for x in X]        

            yield np.array(X)    
# Implement One Cycle Policy Algorithm in the Keras Callback Class



from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

from keras.losses import binary_crossentropy

from keras.metrics import binary_accuracy

from keras import backend as K

from keras.callbacks import *



class CyclicLR(keras.callbacks.Callback):

    

    def __init__(self,base_lr, max_lr, step_size, base_m, max_m, cyclical_momentum):

 

        self.base_lr = base_lr

        self.max_lr = max_lr

        self.base_m = base_m

        self.max_m = max_m

        self.cyclical_momentum = cyclical_momentum

        self.step_size = step_size

        

        self.clr_iterations = 0.

        self.cm_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}

        

    def clr(self):

        

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        

        if cycle == 2:

            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)          

            return self.base_lr-(self.base_lr-self.base_lr/100)*np.maximum(0,(1-x))

        

        else:

            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0,(1-x))

    

    def cm(self):

        

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        

        if cycle == 2:

            

            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1) 

            return self.max_m

        

        else:

            x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

            return self.max_m - (self.max_m-self.base_m)*np.maximum(0,(1-x))

        

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())

            

        if self.cyclical_momentum == True:

            if self.clr_iterations == 0:

                K.set_value(self.model.optimizer.momentum, self.cm())

            else:

                K.set_value(self.model.optimizer.momentum, self.cm())

            

            

    def on_batch_begin(self, batch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)

        

        if self.cyclical_momentum == True:

            self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())

        

        if self.cyclical_momentum == True:

            K.set_value(self.model.optimizer.momentum, self.cm())

            
# Define Ony Cycle Policy parameters and train model

########################################################################################

from keras.optimizers import Adam, SGD

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



# CLR parameters



batch_size = 64

epochs = 10

max_lr = 0.0005

base_lr = max_lr/10

max_m = 0.98

base_m = 0.85



cyclical_momentum = True

augment = True

cycles = 2.35



iterations = round(len(train_list)/batch_size*epochs)

iterations = list(range(0,iterations+1))

step_size = len(iterations)/(cycles)





model.compile(loss='binary_crossentropy', optimizer=SGD(0.0000001), metrics=['accuracy'])



clr =  CyclicLR(base_lr=base_lr,

                max_lr=max_lr,

                step_size=step_size,

                max_m=max_m,

                base_m=base_m,

                cyclical_momentum=cyclical_momentum)

    

callbacks = [clr,

            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',mode='min',verbose=1,save_best_only=True)]



history = model.fit_generator(train_gen(train_list, id_label_map, batch_size, augment = augment),

                              validation_data=train_gen(val_list, id_label_map, batch_size, augment = True),

                              epochs = epochs,

                              steps_per_epoch = len(train_list) // batch_size + 1,

                              validation_steps = len(val_list) // batch_size + 1,

                              callbacks=callbacks,

                              verbose = 1)
# Plot Learning Rate

import matplotlib.pyplot as plt

plt.plot(clr.history['iterations'], clr.history['lr'])

plt.xlabel('Training Iterations')

plt.ylabel('Learning Rate')

plt.title("One Cycle Policy")

plt.show()
# Plot momentum

import matplotlib.pyplot as plt

plt.plot(clr.history['iterations'], clr.history['momentum'])

plt.xlabel('Training Iterations')

plt.ylabel('Momentum')

plt.title("One Cycle Policy")

plt.show()
# Plot losses

val_loss = history.history['val_loss']

loss = history.history['loss']

plt.plot(range(len(val_loss)),val_loss,'c',label='Validation loss')

plt.plot(range(len(loss)),loss,'m',label='Train loss')



plt.title('Training and validation losses')

plt.legend()

plt.xlabel('epochs')

plt.show()