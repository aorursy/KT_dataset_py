#importing required libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

# import plotly.express as px ## gives weired result during model.fit() hence not used



import tensorflow as tf



from keras import Sequential #to create the model arcitecture

from keras.applications import VGG16,VGG19,ResNet50,ResNet101,Xception,InceptionV3,MobileNet,MobileNetV2 #all pretrained models 

#                                     Follow : https://keras.io/api/applications/ for all documentation

from keras.layers import Dense, Input, Dropout, Flatten 

from keras.optimizers import Adam,SGD #optimizers

from keras.callbacks import ReduceLROnPlateau #to reduce learning rate

from keras.preprocessing.image import ImageDataGenerator #Image data generator to generate data of various specification to del with overfitting







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#to check if gpu is available or not

print('GPU is available !!!' if tf.config.list_physical_devices('GPU') else 'GPU is not available')



#GPU 

tf.config.list_physical_devices('GPU')
#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator



datagen_train = ImageDataGenerator(rescale = 1.0/255,  # Ar RGB colors are presented in 0-155 range (1 pixel = 8 bits, since each bit can be 1 or 0, 8 bits info 2^8 = 256 , 0-255 , total 256)

                            horizontal_flip = True,

                            vertical_flip = True,

                             zoom_range = 0.2,

                             shear_range = 0.2,

                             width_shift_range = 0.2,

                             height_shift_range = 0.2,

                             fill_mode = 'nearest'

                             

                            ) 



datagen_val = ImageDataGenerator(rescale = 1.0/255 ) # we dont have other parameters as the model will predict on these images during vaidation



datagen_pred = ImageDataGenerator(rescale = 1.0/255 ) # we dont have other parameters as the model will predict on these images during prediction



train_DIR = "/kaggle/input/intel-image-classification/seg_train/seg_train/"

val_DIR = "/kaggle/input/intel-image-classification/seg_test/seg_test/"

pred_DIR = "/kaggle/input/intel-image-classification/seg_pred/" 



""" we are not using seg_pred/seg_pred/ as datagen requires images to be present inside a folder of certain label.

But in target folder we have all images inside a single folder 'seg_pred'. hence we are using seg_pred instead of seg_pred/seg_pred/



"""

# Follow this link : https://studymachinelearning.com/keras-imagedatagenerator-with-flow_from_directory/

# Follow this link : https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

                                                            



batch_size = 32

image_size = (150,150) #we are converting all images from the directory to this shape as our models need input of constant shape



train_datagen = datagen_train.flow_from_directory(train_DIR,

                                                 batch_size = batch_size,

                                                 target_size = image_size,

                                                  class_mode = 'categorical',

                                                  color_mode = 'rgb',

#                                                   seed = 101

                                                 )

val_datagen = datagen_val.flow_from_directory(val_DIR,

                                             batch_size = batch_size,

                                             target_size = image_size,

                                             class_mode = 'categorical',

                                             color_mode ='rgb',

#                                              seed = 110

                                             )

pred_datagen = datagen_pred.flow_from_directory(pred_DIR,

                                               batch_size = 1, # as we want all images in one batch during prediction

                                               target_size = image_size,

                                                class_mode = None, # to return only image

                                               color_mode ='rgb',

#                                                seed = 150

                                               ) 

#class lables in train dataset

train_datagen.class_indices

# validation classes

val_datagen.class_indices
#mapping encoded values to class labels

labels = train_datagen.class_indices

labels = dict((v,k) for k,v in labels.items())

labels # now we have encoded values as keys and class name as valaues. This helps during decoding the predicition
#plotting some images from image generator https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/



fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(16,16))





for i in range (5):

    

    image = next(train_datagen)[0][0] # getting images

    

    image = np.squeeze(image) # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image

    

    ax[i].imshow(image)

    ax[i].axis('off')
# Hyperparameters:





input_shape = (150,150,3)

batch_size = 32

lr = 0.001

n_class = 6

epochs =20

adam = Adam(lr = lr, beta_1 = 0.9, beta_2 =0.999,amsgrad =False,epsilon =1e-7)



#ReduceLROnPlateau to reduce LR : https://keras.io/api/callbacks/reduce_lr_on_plateau/



lrr = ReduceLROnPlateau(monitor = 'val_acc',

                       patience = 1,

                       factor =0.25,

                        min_lr = 0.000003,

                        verbose =1

                       )
#Creating early stopping callback

from  keras.callbacks import EarlyStopping

early_stopping =EarlyStopping(monitor = 'val_accuracy', patience=3) #stop the training process if there is no change in val_accuracy for 3 rounds
#instatiating the model

vgg16 = VGG16(include_top = False,input_shape = input_shape,

                  weights='imagenet',

                  classes = n_class)



# #we are freezing all layers and training only fully connected layers

# for layer in vgg16.layers:

#     layer.trainable =False
#creating a function to build the FC by taking the base model and return the final model



def build_model(base_modelx):

    

    for layer in base_modelx.layers:

        layer.trainable = False

    

    model = Sequential(base_modelx)

    model.add(Flatten())

    model.add(Dense(1024,activation ='relu'))

#     model.add(Dropout(0.3))

    model.add(Dense(512,activation = 'relu'))

    model.add(Dropout(0.3))

    model.add(Dense(256,activation = 'relu'))

    model.add(Dropout(0.2))

    model.add(Dense(128,activation = 'relu'))

    model.add(Dropout(0.15))

    model.add(Dense(n_class,activation='softmax'))

    

    print(model.summary())

    

    model.compile(loss = 'categorical_crossentropy',optimizer = adam,metrics =['acc'])

    

    return model

    

    
model = build_model(vgg16)

#itting the model

model.fit(train_datagen,

          epochs = epochs,

          validation_data = val_datagen,

          verbose =1,

          callbacks =[lrr,early_stopping]

         )
# model.history.history
# len(model.history.history['acc'])
#Ploting acc and loss

results = pd.DataFrame({'epochs':list(range(1,epochs+1)),'Training_acc':model.history.history['acc'],'Validation_acc':model.history.history['val_acc'],

                      'Training_loss':model.history.history['loss'],'Validation_loss':model.history.history['val_loss']})



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_acc', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_acc', data = results, color='blue' )

plt.title('Training Accuracy vs Validation Accuracy')

plt.show()



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_loss', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_loss', data = results, color='blue' )

plt.title('Training Loss vs Validation Loss')

plt.show()
vgg19 = VGG19(include_top = False, weights ='imagenet', input_shape = input_shape,

                   classes =n_class)



  

model = build_model(vgg19)
#itting the model

model.fit(train_datagen,

          epochs = epochs,

          validation_data = val_datagen,

          verbose =1,

          callbacks =[lrr,early_stopping]

         )
#Ploting acc and loss

results = pd.DataFrame({'epochs':list(range(1,epochs+1)),'Training_acc':model.history.history['acc'],'Validation_acc':model.history.history['val_acc'],

                      'Training_loss':model.history.history['loss'],'Validation_loss':model.history.history['val_loss']})



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_acc', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_acc', data = results, color='blue' )

plt.title('Training Accuracy vs Validation Accuracy')

plt.show()



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_loss', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_loss', data = results, color='blue' )

plt.title('Training Loss vs Validation Loss')

plt.show()
resnet50 = ResNet50(include_top = False, weights ='imagenet',

                   input_shape = input_shape,

                   classes = n_class)

model = build_model(resnet50)
model.fit(train_datagen, epochs = epochs,

         validation_data = val_datagen,

         verbose =1,

         callbacks =[lrr,early_stopping])
#Ploting acc and loss

results = pd.DataFrame({'epochs':list(range(1,epochs+1)),'Training_acc':model.history.history['acc'],'Validation_acc':model.history.history['val_acc'],

                      'Training_loss':model.history.history['loss'],'Validation_loss':model.history.history['val_loss']})



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_acc', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_acc', data = results, color='blue' )

plt.title('Training Accuracy vs Validation Accuracy')

plt.show()



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_loss', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_loss', data = results, color='blue' )

plt.title('Training Loss vs Validation Loss')

plt.show()
resnet101 = ResNet101(include_top = False, weights = 'imagenet',

                      input_shape = input_shape,

                     classes =n_class)

model = build_model(resnet101)
model.fit(train_datagen,epochs = epochs,

         validation_data = val_datagen,

         verbose =1,

         callbacks = [lrr,early_stopping])
#Ploting acc and loss

results = pd.DataFrame({'epochs':list(range(1,epochs+1)),'Training_acc':model.history.history['acc'],'Validation_acc':model.history.history['val_acc'],

                      'Training_loss':model.history.history['loss'],'Validation_loss':model.history.history['val_loss']})



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_acc', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_acc', data = results, color='blue' )

plt.title('Training Accuracy vs Validation Accuracy')

plt.show()



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_loss', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_loss', data = results, color='blue' )

plt.title('Training Loss vs Validation Loss')

plt.show()
xception = Xception(include_top = False, weights = 'imagenet',

                    input_shape = input_shape,

                    classes = n_class)



model = build_model(xception)
model.fit(train_datagen, epochs = epochs,

         validation_data = val_datagen,

         verbose =1,

         callbacks = [lrr,early_stopping])
#Ploting acc and loss

results = pd.DataFrame({'epochs':list(range(1,epochs+1)),'Training_acc':model.history.history['acc'],'Validation_acc':model.history.history['val_acc'],

                      'Training_loss':model.history.history['loss'],'Validation_loss':model.history.history['val_loss']})



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_acc', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_acc', data = results, color='blue' )

plt.title('Training Accuracy vs Validation Accuracy')

plt.show()



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_loss', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_loss', data = results, color='blue' )

plt.title('Training Loss vs Validation Loss')

plt.show()
inception = InceptionV3(include_top= False, weights = 'imagenet',

                       input_shape = input_shape,

                       classes = n_class) 

model = build_model(inception)
model.fit(train_datagen, epochs = epochs,

           validation_data = val_datagen,

           verbose = 1,

           callbacks = [lrr,early_stopping])
#Ploting acc and loss

results = pd.DataFrame({'epochs':list(range(1,epochs+1)),'Training_acc':model.history.history['acc'],'Validation_acc':model.history.history['val_acc'],

                      'Training_loss':model.history.history['loss'],'Validation_loss':model.history.history['val_loss']})



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_acc', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_acc', data = results, color='blue' )

plt.title('Training Accuracy vs Validation Accuracy')

plt.show()



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_loss', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_loss', data = results, color='blue' )

plt.title('Training Loss vs Validation Loss')

plt.show()
mobilenetv2 = MobileNetV2(include_top= False, weights = 'imagenet',

                       input_shape = input_shape,

                       classes = n_class) 

model = build_model(mobilenetv2)
model.fit(train_datagen, epochs = epochs,

           validation_data = val_datagen,

           verbose = 1,

           callbacks = [lrr,early_stopping])
#Ploting acc and loss

results = pd.DataFrame({'epochs':list(range(1,epochs+1)),'Training_acc':model.history.history['acc'],'Validation_acc':model.history.history['val_acc'],

                      'Training_loss':model.history.history['loss'],'Validation_loss':model.history.history['val_loss']})



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_acc', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_acc', data = results, color='blue' )

plt.title('Training Accuracy vs Validation Accuracy')

plt.show()



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_loss', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_loss', data = results, color='blue' )

plt.title('Training Loss vs Validation Loss')

plt.show()
mobilenet = MobileNet(include_top= False, weights = 'imagenet',

                       input_shape = input_shape,

                       classes = n_class) 

model = build_model(mobilenet)
model.fit(train_datagen, epochs = epochs,

           validation_data = val_datagen,

           verbose = 1,

           callbacks = [lrr,early_stopping])
#Ploting acc and loss

results = pd.DataFrame({'epochs':list(range(1,epochs+1)),'Training_acc':model.history.history['acc'],'Validation_acc':model.history.history['val_acc'],

                      'Training_loss':model.history.history['loss'],'Validation_loss':model.history.history['val_loss']})



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_acc', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_acc', data = results, color='blue' )

plt.title('Training Accuracy vs Validation Accuracy')

plt.show()



plt.figure(figsize=(12,5))

sns.lineplot(x = 'epochs', y ='Training_loss', data = results, color='r' )

sns.lineplot(x = 'epochs', y ='Validation_loss', data = results, color='blue' )

plt.title('Training Loss vs Validation Loss')

plt.show()
#Prediction:

preds = model.predict(pred_datagen)

preds[0:5,:]
# preds[1]
#get the indices

pred_class_indices = np.argmax(preds,axis=1)

pred_class_indices[0:26]
#get the name of the label

label_names = [labels[k] for k in pred_class_indices]

label_names[0:26]
# plt.imshow(np.squeeze(pred_datagen[19]))
# lets check those images

n=25



# setup the figure 

plt.figure(figsize=(20,20))



for i in range(n):

#     print(i)

    ax = plt.subplot(5, 5, i+1)

    plt.imshow(np.squeeze(pred_datagen[i]))