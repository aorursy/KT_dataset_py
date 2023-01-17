import keras

from keras.models import Model

from keras.callbacks import EarlyStopping ,ModelCheckpoint

from keras.layers import Dense,GlobalAveragePooling2D,ReLU,Dropout

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers

from keras.applications.xception import Xception 

from keras import optimizers

from keras.models import load_model

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

import tensorflow



import numpy as np

import cv2

import time

import matplotlib.pyplot as plt





import os

print(os.listdir("../input/fcc_data_0717"))



%matplotlib inline
dir_path_train = '../input/fcc_data_0717/train'

dir_path_val = '../input/fcc_data_0717/valid'

dir_path_train_good = '../input/fcc_data_0717/train/good'

dir_path_train_bad = '../input/fcc_data_0717/train/bad'

dir_path_valid_good = '../input/fcc_data_0717/valid/good'

dir_path_valid_bad = '../input/fcc_data_0717/valid/bad'





epochs = 40

ft_epochs = 100

inputshape = (299,299)
valid_size = len(os.listdir(dir_path_valid_good)) + len(os.listdir(dir_path_valid_bad))

print(valid_size)
def model_init(): 

      

    

    base_model = Xception(weights='imagenet',pooling = 'avg',include_top=False)

    for layer in base_model.layers[:-17]:

        layer.trainable = False

    

    x = base_model.output

    x = Dropout(0.5)(x)

    x = Dense(1024,kernel_regularizer = regularizers.l2(0.05))(x)

    x = BatchNormalization()(x)

    x = ReLU()(x)

    x = Dropout(0.5)(x)

    x = Dense(32,kernel_regularizer = regularizers.l2(0.05))(x)

#     x = BatchNormalization()(x)

#     x = ReLU()(x)

#     x = Dropout(0.2)(x)

#     x = Dense(256,kernel_regularizer = regularizers.l2(0.005))(x)

#     x = BatchNormalization()(x)

#     x = ReLU()(x)

#     x = Dropout(0.2)(x)

#     x = Dense(128,activation = 'relu',kernel_regularizer = regularizers.l2(0.005))(x)

    prediction = Dense(1,activation='sigmoid')(x)



    model = Model(inputs=base_model.input, outputs=prediction)

    

    model.summary()



    return model
model1 = model_init()

def model_fit(model,path_train,path_val,train_batch_size,val_batch_size):

#     early_stopping = EarlyStopping(monitor = 'val_loss',patience = 10)

#     checkpoint = ModelCheckpoint(filepath = 'model.h5',monitor='val_acc')

#     callbacks = [early_stopping,checkpoint]

    

    train_image_generator = ImageDataGenerator(rescale = 1./255,

                                              rotation_range = 20,

                                              width_shift_range = 0.2,

                                              height_shift_range = 0.2,

                                              shear_range=0.2,

                                              zoom_range=0.2,

                                              horizontal_flip = True)

    

    valid_image_generator = ImageDataGenerator(rescale = 1./255)

    

    train_generator = train_image_generator.flow_from_directory(path_train,

                                                    target_size = inputshape,

                                                    batch_size = train_batch_size,

                                                    class_mode = 'binary',

                                                    shuffle = True)

    

    

    valid_generator = valid_image_generator.flow_from_directory(path_val,

                                                    target_size = inputshape,

                                                    batch_size = 32,

                                                    class_mode = 'binary',

                                                    shuffle = True)

    

    adam = optimizers.Adam(lr=0.001)

    model.compile(optimizer=adam,

                  loss='binary_crossentropy',

                 metrics=['accuracy'])

    

    print('Training Started!')

    start = time.time()

    history = model.fit_generator(train_generator,

                                  epochs = epochs,

                                  verbose = 2,

                                  validation_data = valid_generator,

                                  steps_per_epoch = 100,

                                 validation_steps = 25)

    print('Training Finished')

    end = time.time()

    print(f'Total fiting time:{round(end - start,2)}s')

    

    plt.figure(figsize=(12, 9))

    valacc,= plt.plot(model1.history.history['val_acc'])

    trainacc,= plt.plot(model1.history.history['acc'])

    plt.xlabel('epoch')

    plt.ylabel('Accuracy')

    plt.yticks(np.arange(0.5,1,0.1))

    plt.ylim(0.5, 1)

    plt.legend([valacc,trainacc],['Validation','Train'])

    plt.title('LEARNING CURVE')

    

    return model
# SVG(model_to_dot(model1).create(prog='dot', format='svg'))
model1 = model_fit(model1,dir_path_train,dir_path_val,32,32)

model1.save('fcc_model_0717_3.h5')
def finetune_setup(model,layers):

    """

    top layers plus last several base layers to be hold untouched

    

    layers = 32 : block 13 and 14 plus top layer

    layers = 13 : block 14 plus top layer

    """

    

    for layer in model.layers[:(len(model.layers) - layers)]:

        layer.trainable = False

    for layer in model.layers[(len(model.layers) - layers):]:

        layer.trainable = True

    

    model.compile(loss = 'binary_crossentropy',

                 optimizer = optimizers.SGD(lr=1e-4,momentum = 0.9),

                 metrics = ['accuracy']

                 )

    

    

def fine_tune_model(model,path_train,path_val,train_batch_size,val_batch_size):

    

    train_image_generator = ImageDataGenerator(rescale = 1./255,

                                              rotation_range = 20,

                                              width_shift_range = 0.2,

                                              height_shift_range = 0.2,

                                              shear_range=0.2,

                                              zoom_range=0.2,

                                              horizontal_flip = True)

    

    valid_image_generator = ImageDataGenerator(rescale = 1./255)

    

    train_generator = train_image_generator.flow_from_directory(path_train,

                                                    target_size = inputshape,

                                                    batch_size = train_batch_size,

                                                    class_mode = 'binary',

                                                    shuffle = True)

    

    

    valid_generator = valid_image_generator.flow_from_directory(path_val,

                                                    target_size = inputshape,

                                                    batch_size = val_batch_size,

                                                    class_mode = 'binary',

                                                    shuffle = True)

    





    print('Training Starts!')

    start = time.time()

    history = model.fit_generator(train_generator,

                       epochs = ft_epochs,

                        verbose = 2,

                        validation_data = valid_generator,

                        validation_steps = len(valid_generator),

                        steps_per_epoch = len(train_generator))

    end = time.time()

    print('Training Finished!')

    print(f'Total Time:{round(end - start,2)}s')

    

    

     

                        

    plt.figure(figsize=(12, 9))

    valacc,= plt.plot(model.history.history['val_acc'])

    trainacc,= plt.plot(model.history.history['acc'])

    plt.xlabel('epoch')

    plt.ylabel('Accuracy')

    plt.yticks(np.arange(0.5,1,0.1))

    plt.ylim(0.5, 1)

    plt.legend([valacc,trainacc],['Validation','Train'])

    plt.title('LEARNING CURVE')     

    

    return model
# finetune_setup(model1,32)

# model1_ft = fine_tune_model(model1,dir_path_train,dir_path_val,100,300)

# model1_ft.save('fcc_model_0716_1.h5')