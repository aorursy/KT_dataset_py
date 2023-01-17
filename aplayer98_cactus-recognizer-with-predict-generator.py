import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,cv2

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

from keras import layers,models

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import seaborn as sns

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential

from keras.applications.vgg16 import VGG16



import numpy as np

import math



from keras.layers import Dropout, Flatten,Activation

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop



#System

import os

print(os.listdir("../input"))
#input datasets

train_dir = '../input/aerial-cactus-identification/train/train'

test_dir = '../input/aerial-cactus-identification/test/test'



labels = pd.read_csv('../input/aerial-cactus-identification/train.csv')



x_train = labels.id

y_train = labels.has_cactus



print('total row and column of data =' + str(labels.shape[0:]))

print('total image with cactus count =',sum(y_train == 1))

labels.head()
#conversion of has_Cactus from int to string so that it can fit train_generator

labels.has_cactus = labels.has_cactus.astype(str)



#specify details of image generator

train_datagen = ImageDataGenerator(

        #normalize all image

        rescale=1./255,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        vertical_flip = True,

        fill_mode='nearest')



#use .flowfromDataFrame but not .flowfromDirectory as images are clustered in one folder

#dataframe = csv file

train_generator = train_datagen.flow_from_dataframe(

    dataframe = labels[:13500],

    directory = train_dir,

    x_col = 'id',

    y_col = 'has_cactus',

    target_size = (128,128),

    color_mode = 'rgb',

    class_mode = 'binary')



val_datagen = ImageDataGenerator(rescale = 1./255)



val_generator = val_datagen.flow_from_dataframe(

    dataframe = labels[13500:],

    directory = train_dir,

    x_col = 'id',

    y_col = 'has_cactus',

    target_size = (128,128),

    color_mode = 'rgb',

    class_mode = 'binary')
#set base_model as transfer learning model



base_model = VGG16(include_top = False,weights =None,input_shape =(128,128,3))

#manually added weights as failed to download from Kaggle

base_model.load_weights('../input/trans-learn-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
val_generator = val_datagen.flow_from_dataframe(

    dataframe = labels[13500:],

    directory = train_dir,

    x_col = 'id',

    y_col = 'has_cactus',

    target_size = (128,128),

    color_mode = 'rgb',

    class_mode = 'binary')

base_model.summary()
#Declare sequential for transfer training

model = Sequential()



#Add basemodel and 1 final layers

model.add(base_model)

model.add(Flatten())

model.add(Dense(256,activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation = 'sigmoid'))
model.summary()
for i in range (len(base_model.layers)):

    print (i,base_model.layers[i])
#Unfreezing last 2 layers of VGG16 model

for layer in base_model.layers[11:]:

    layer.trainable=True

for layer in base_model.layers[0:11]:

    layer.trainable=False

print('Unfreezed base model')
epochs=8

batch_size=128



#Using learning rate annealer

red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1)
#Compile all layers

#loss is binary_crossentropy as the problem only involve 1 and 0 ( binary ）

#optimizer alternative : optimizers.rmsprop()

model.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy',metrics=['accuracy'])



print('model compiled')
#Fit the model

History = model.fit_generator(train_generator, callbacks = [red_lr],

                              epochs = epochs, validation_data = val_generator, validation_steps = 50,

                              verbose = 1, steps_per_epoch= math.ceil(labels.shape[0]/ batch_size))
#Visualizing the result of accuracy

plt.plot(History.history['acc'])

plt.plot(History.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
#Visualizing the result of loss

plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
#save model using pickle method

import pickle



#save the model

model_file = "model.sav"

with open(model_file,mode='wb') as model_f:

    pickle.dump(model,model_f)
test_label=pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')



test_datagen = ImageDataGenerator(rescale = 1./255)



test_generator = test_datagen.flow_from_dataframe(

    dataframe = test_label,

    directory = test_dir,

    x_col = 'id',

    y_col = 'has_cactus',

    target_size = (128,128),

    color_mode = 'rgb',

    class_mode = 'other',

    shuffle = False)



predict = model.predict_generator(test_generator,steps = test_label.shape[0],verbose = 1)
#Load the model

#with open(model_file,mode='rb') as model_f:

#model = pickle.load(model_f)
#Code for submission

print(predict)

y_submit = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

count = 0

while (count <= y_submit.shape[0]):

    y_submit.has_cactus[count] = predict[count]

    count += 1

    if(count % 100 == 0):

        print('done 100 copies'+str(count))

    if(count % 4000 == 0):

        break

y_submit.to_csv('submission.csv',index=False)
y_submit.head()