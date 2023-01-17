import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
import os
from xml.etree import cElementTree as ElementTree
from sys import stdout
from time import time
from PIL import Image
from display_box import dataset
from display_box import XmlDictConfig,XmlListConfig,dataset

import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Conv2D,Input,MaxPooling2D,Flatten,MaxPool2D,Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
height , width , channels = 225, 225 , 3
n_classes = 4
n_coordinate = 4
img_input = Input(shape=(height,width,channels))
x = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(img_input)
x = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.1)(x)
x = Conv2D(filters=128,kernel_size=(3,3),activation='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)
x = Conv2D(filters=256,kernel_size=(3,3),activation='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
y = Dense(units=512,activation='sigmoid')(x)
z = Dense(units=1024,activation='sigmoid')(x)
confidences= Dense(units = n_classes , activation='softmax',name='classes')(y)
coordinate = Dense(units = n_coordinate , activation='sigmoid',name='boxes')(z)
model_loc_cls =Model(inputs=img_input, outputs = [coordinate,confidences])
model_loc_cls.summary()
plot_model(model_loc_cls,show_layer_names=True,show_shapes=True,rankdir='TB',expand_nested = True,dpi=75)
#@title Image Data Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def data_generator(TRAINING_DIR,TEST_DIR,TARGET_SIZE,COLOR_MODE,BATCH_SIZE=32,
                     CLASSE_MODE='categorical', SHUFFLE=True,SHUFFLE_TEST = False):
                   
  training_datagen = ImageDataGenerator(rescale=1. / 255)                              
  testing_datagen = ImageDataGenerator(rescale=1. / 255)
  "Takes the path to a directory & generates batches of augmented data."
  train_generator = training_datagen.flow_from_directory(
                                                          TRAINING_DIR,
                                                          target_size=TARGET_SIZE,
                                                          class_mode=CLASSE_MODE,
                                                          color_mode=COLOR_MODE,
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=SHUFFLE
                                                        )
  "Takes the path to a directory & generates batches of augmented data."
  test_generator = testing_datagen.flow_from_directory(
                                                          TEST_DIR,
                                                          target_size=TARGET_SIZE,
                                                          class_mode=CLASSE_MODE,
                                                          color_mode=COLOR_MODE,
                                                          batch_size= BATCH_SIZE,
                                                          shuffle=SHUFFLE_TEST
                                                      )
  return train_generator,test_generator
import tensorflow as tf
mobile=tf.keras.applications.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
    )
for layer in mobile.layers[:-2]:
  layer.trainable = False
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Conv2D,Input,MaxPooling2D,Flatten,MaxPool2D,Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
height , width , channels = 225, 225 , 3
n_classes = 4
n_coordinate = 4
img_input = mobile.input
x = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(img_input)
x = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.1)(x)
x = Conv2D(filters=128,kernel_size=(3,3),activation='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)
x = Conv2D(filters=256,kernel_size=(3,3),activation='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
y= mobile.get_layer(index=-3).output
z =Flatten()(y)

coordinate = Dense(units = n_coordinate , activation='sigmoid',name='boxes')(x)
confidences= Dense(units = n_classes , activation='softmax',name='classes')(z)

model_loc_cls =Model(inputs=img_input, outputs = [coordinate,confidences])
plot_model(model_loc_cls,show_layer_names=True,show_shapes=True,rankdir='TB',expand_nested = True,dpi=75)
model_loc_cls.compile(
                      optimizer = 'adam', loss = {'classes':'categorical_crossentropy',
                                                  'boxes':'MeanAbsoluteError'},
                      metrics= {'classes':'accuracy','boxes':custom.iou_metric},
                      loss_weights={'classes':1.0,'boxes':10.0}
                      )
objet = dataset(folder_path ='/kaggle/input/microcontroller-detection/Microcontroller Detection/train/',target_shape=(224,224))
objet_test= dataset(folder_path ='/kaggle/input/microcontroller-detection/Microcontroller Detection/test/',
                    target_shape=(224,224))

adapt_data_to_image_generator('/content/drive/My Drive/DATASETS/Microcontroller Detection/train_image_2',xml_objet =objet)
#adapt_data_to_image_generator('/content/drive/My Drive/DATASETS/Microcontroller Detection/test_image_2',xml_objet =objet_test)
import tensorflow
import numpy as np
'''
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor ='val_loss', mode='min', verbose=0, patience=5)
mc = ModelCheckpoint('/content/drive/My Drive/best_model.h5', monitor='boxes_iou_metric', mode='max', verbose=0, save_best_only=True)
'''
my_histo = []
def tt(BATCH_SIZE):
  return dd(BATCH_SIZE)[0],dd(BATCH_SIZE)[1]
  
BATCH_SIZE = 40;tuple_train,tuple_test = tt(BATCH_SIZE)

for e in range(50):
  print('epochs ', e)

  if e == 32: 
    BATCH_SIZE = 30;tuple_train,tuple_test = tt(BATCH_SIZE)
  if e == 45 :
     BATCH_SIZE = 15;tuple_train,tuple_test = tt(BATCH_SIZE)
  if e == 50 : 
    BATCH_SIZE = 10;tuple_train,tuple_test = tt(BATCH_SIZE) 
  if e == 60 : 
    BATCH_SIZE = 5;tuple_train,tuple_test = tt(BATCH_SIZE)
  datagen =tuple_train
  batches_per_epoch = datagen.samples // datagen.batch_size + (datagen.samples % datagen.batch_size > 0)

  
  for i in range(batches_per_epoch):
      curent_batch_train = next(datagen)
      curent_batch_test = next(tuple_test)

      def small(datagen):
        current_index = ((datagen.batch_index-1) * datagen.batch_size)               
        if current_index < 0:
            if datagen.samples % datagen.batch_size > 0:
                return max(0,datagen.samples - datagen.samples % datagen.batch_size)
            else:
                return max(0,datagen.samples - datagen.batch_size)
        else:
          return current_index 

      current_index = small(datagen)
      current_index_test =small(tuple_test)

      #Train
      index_array = datagen.index_array[current_index:current_index + datagen.batch_size].tolist()
      img_file_name = [datagen.filenames[idx].split('/')[-1]  for idx in index_array]
      cls_one_hot_train=curent_batch_train[1]
      #Test
      index_array_test = tuple_test.index_array[current_index_test:current_index_test + tuple_test.batch_size].tolist()
      img_file_name_test = [tuple_test.filenames[idx].split('/')[-1] for idx in index_array_test]        
      cls_one_hot_test=curent_batch_test[1] 

      annot_train = [filename.split('/')[-1] for filename in objet.list_dir_files]
      annot_test = [filename.split('/')[-1] for filename in objet_test.list_dir_files]
      boxes_train = np.array( [objet.resized_coordinate [annot_train.index(name)]  for name in img_file_name] ) / 225.0
      boxes_test = np.array( [objet_test.resized_coordinate[annot_test.index(name)] for name in img_file_name_test] ) / 225.0
      # print(model_loc_cls.train_on_batch(x = curent_batch_train , y= [boxes_train, cls_one_hot_train],reset_metrics=False))
      histo = model_loc_cls.fit(x = curent_batch_train , y= [boxes_train, cls_one_hot_train],
                          validation_data=(curent_batch_test,[ boxes_test ,cls_one_hot_test ]),
                          epochs = 1,
                          batch_size = datagen.batch_size,
                          initial_epoch = 0,
                          callbacks=[mc]
                        )      
      if i == 0:
        myhisto = histo.history
      else:
        for key in list(histo.history.keys()):
          myhisto[key].append(histo.history[key][0])

import tensorflow
class file:
    
        def save_model(architecture,name_file,save_weight = True,dictn=True):

            # serialize model(Architecture) to JSON
            model_json = architecture.to_json()
            with open(name_file+'_model'+".json", "w") as json_file:
                print('Save the architecture of model to disk ... ')
                json_file.write(model_json)
            
            # serialize weights to HDF5                
            if save_weight:
                
              print("Save weights to disk ... ")
              architecture.save_weights(name_file+'_weights'+".h5")
        
        
        def load_model(path_model,path_weight=None):

            # load json and create model
            
            json_file = open(path_model, 'r')
            print("Load architecture model from disk ... ")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = tensorflow.keras.models.model_from_json(loaded_model_json)
            
            # load weights into new model
  
            if path_weight:
                print("Loaded weights model from disk ...")
                loaded_model.load_weights(path_weight)
            
            return loaded_model
file.save_model(model_loc_cls,name_file='micro_transfer_learning_regression')
model = file.load_model(path_model='/kaggle/input/modelmicro/micro_transfer_learning_regression_model.json',
                        path_weight= '/kaggle/input/modelmicro/micro_transfer_learning_regression_weights.h5')
objet = dataset(folder_path ='/kaggle/input/microcontroller-detection/Microcontroller Detection/train/',target_shape=(224,224))
objet.show_boxes(30,model=model,subplot=(6,5,1))
#prediction red
#true boxes green
objet_test= dataset(folder_path ='/kaggle/input/microcontroller-detection/Microcontroller Detection/test/',
                    target_shape=(224,224))
objet_test.show_boxes(7,model=model,subplot=(2,5,1))