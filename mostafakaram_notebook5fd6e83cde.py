# import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import os
import PIL

# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 

train = []
test = []

# os.listdir returns the list of files in the folder, in this case image class names
for i in os.listdir('../input/satimages-1/train'):
  train_class = os.listdir(os.path.join('../input/satimages-1/train', i))
  train.extend(train_class)
  test_class = os.listdir(os.path.join('../input/satimages-1/test', i))
  test.extend(test_class)
    
print("Number of trained data is : ", len(train))
print("Number of tested data is : ", len(test))
fig, axs = plt.subplots(3,3, figsize=(32,32))
count = 0
for i in os.listdir('../input/satimages-1/train'):
  # get the list of images in the particular class
  train_class = os.listdir(os.path.join('../input/satimages-1/train',i))
  # plot 5 images per class
  for j in range(3):
        img=os.path.join('../input/satimages-1/train',i,train_class[j])
        img=PIL.Image.open(img)
        axs[count][j].imshow(img)
        axs[count][j].set_title(i,fontsize=20)
  count+=1

fig.tight_layout()

No_images_per_class = []
Class_name = []
for i in os.listdir('../input/satimages-1/train'):
  train_class = os.listdir(os.path.join('../input/satimages-1/train', i))
  No_images_per_class.append(len(train_class))
  Class_name.append(i)
  print('Number of images in {} = {} \n'.format(i, len(train_class)))
fig1, axs1=plt.subplots()
axs1.pie(No_images_per_class, labels= Class_name, autopct='%1.1f%%')
print(fig1,"   ",axs1)
train_datagen = ImageDataGenerator(
               rescale=1./255,
               zoom_range=0.2,
               validation_split=0.15,
               horizontal_flip=True,
               vertical_flip=True)

# For test datagenerator, we only normalize the data.
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../input/satimages-1/train',
    target_size = (256,256),
    batch_size = 10,
    class_mode = 'categorical',
    subset = 'training'
)

validation_generator = train_datagen.flow_from_directory(
    '../input/satimages-1/train',
    target_size = (256,256),
    batch_size = 10,
    class_mode = 'categorical',
    subset = 'validation'
)

test_generator = test_datagen.flow_from_directory(
    '../input/satimages-1/test',
    target_size = (256,256),
    batch_size = 10,
    class_mode = 'categorical',
)
input_shape = (256,256,3)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3,3))(X_input)

# 1 - stage
X = Conv2D(64, (7,7), strides= (2,2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3,3), strides= (2,2))(X)

# 2- stage
X = res_block(X, filter= [64,64,256], stage= 2)

# 3- stage
X = res_block(X, filter= [128,128,512], stage= 3)

# 4- stage
X = res_block(X, filter= [256,256,1024], stage= 4)

# 5- stage
X = res_block(X, filter= [512,512,2048], stage= 5)

# Average Pooling
X = AveragePooling2D((2,2), name = 'Averagea_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dropout(0.4)(X)

X = Dense(3, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)


model = Model( inputs= X_input, outputs = X, name = 'Resnet18')

model.summary()
input_shape = (256,256,3)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3,3))(X_input)

# 1 - stage
X = Conv2D(64, (7,7), strides= (2,2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3,3), strides= (2,2))(X)

# 2- stage
X = res_block(X, filter= [64,64,256], stage= 2)

# 3- stage
X = res_block(X, filter= [128,128,512], stage= 3)

# 4- stage
X = res_block(X, filter= [256,256,1024], stage= 4)

# 5- stage
X = res_block(X, filter= [512,512,2048], stage= 5)

# Average Pooling
X = AveragePooling2D((2,2), name = 'Averagea_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dropout(0.4)(X)

X = Dense(3, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)


model = Model( inputs= X_input, outputs = X, name = 'Resnet18')

model.summary()
model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics=['accuracy'])
# using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="../input/satimages-1/weights_22.hdf5", verbose=1, save_best_only=True)
model.fit_generator(train_generator, steps_per_epoch= train_generator.n // 32, epochs = 2, validation_data= validation_generator, validation_steps= validation_generator.n // 10, callbacks=[checkpointer , earlystopping])
evaluate = model.evaluate_generator(test_generator, steps = test_generator.n // 10, verbose =2)

labels = {0: 'desert', 1: 'plant', 2: 'water'}
# load images and their predictions 

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import cv2

prediction = []
original = []
image = []
count = 0
for i in os.listdir('../input/satimages-1/test'):
    for item in os.listdir(os.path.join('../input/satimages-1/test', i)):
        # code to open the image
        img= PIL.Image.open(os.path.join('../input/satimages-1/test', i, item))
        # resizing the image to (256,256)
        img = img.resize((256, 256))
        # appending image to the image list
        image.append(img)
        # converting image to array
        img = np.asarray(img, dtype = np.float32)
        # normalizing the image
        img = img / 255
        # reshaping the image into a 4D array
        img = img[...,:3]
        #-------------------------------------
        img = img.reshape(-1, 256, 256, 3)
        # making prediction of the model
        predict = model.predict(img)
        # getting the index corresponding to the highest value in the prediction
        predict = np.argmax(predict)
        # appending the predicted class to the list
        prediction.append(labels[predict])
        # appending original class to the list
        original.append(i)
# Get the test accuracy 
score = accuracy_score(original, prediction)
print("Test Accuracy : {}".format(score))

# visualize the results
import random
fig = plt.figure(figsize = (100,100))
for i in range(20):
    j = random.randint(0, len(image))
    fig.add_subplot(20, 1, i+1)
    plt.xlabel("Prediction: " + prediction[j] +"   Original: " + original[j])
    plt.imshow(image[j])
fig.tight_layout()
plt.show()
