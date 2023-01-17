!git clone https://github.com/sudofix/sat-classification
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
import cv2
from PIL import Image 
# from jupyterthemes import jtplot
# jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 

os.chdir('./sat-classification')
os.listdir()
# Check the number of images in training, validation and test dataset

def round(x):
    return int (x*100) / 100 
train = []
test = []

# os.listdir returns the list of files in the folder, in this case image class names
for i in os.listdir('./train'):
    train_class = os.listdir(os.path.join('train', i))
    train.extend(train_class)
    test_class = os.listdir(os.path.join('test', i))
    test.extend(test_class)

print("Number of training images : {} ({}%) \nNumber of testing images {} ({}%)".format(len(train) ,round((len(train)/(len(train)+len(test)) )*100) ,  len(test) , round((len(test)/(len(train)+len(test)) )*100) ))


os.listdir('./train')
# Visualize the images in the dataset

fig, axs = plt.subplots(3,5, figsize=(32,32))
row = 0
for i in os.listdir('./train'):
    # get the list of images in the particular class
    train_class = os.listdir(os.path.join('train',i))
    # plot 5 images per class
    for j in range(5):
        img = os.path.join('train' , i , train_class[j])
        img = PIL.Image.open(img)
        axs[row][j].imshow(img)
        axs[row][j].set_title(i, fontSize = 30)
    row+=1
fig.tight_layout()

No_of_images_per_class  = []
Class_name = []
for i in os.listdir('./train'):
    train_class = os.listdir(os.path.join('train' , i ))
    No_of_images_per_class.append(len(train_class))
    Class_name.append(i)
    print("Number of images in {} = {}\n".format(i , len(train_class)))
fig1 , ax1 = plt.subplots()
ax1.pie(No_of_images_per_class , labels = Class_name , autopct = '%1.1f%%')
plt.show
No_of_images_per_class  = []
Class_name = []
for i in os.listdir('./test'):
    train_class = os.listdir(os.path.join('test' , i ))
    No_of_images_per_class.append(len(train_class))
    Class_name.append(i)
    print("Number of images in {} = {}\n".format(i , len(train_class)))
fig1 , ax1 = plt.subplots()
ax1.pie(No_of_images_per_class , labels = Class_name , autopct = '%1.1f%%')
plt.show
train_datagen = ImageDataGenerator(
                rescale = 1./255,
                brightness_range=[0.50 , 1.0],
                rotation_range = 20,
                shear_range=0.25,
                zoom_range=[0.3 , 0.5],
                channel_shift_range= 0.3,
                validation_split = 0.15,
                horizontal_flip = True,
                vertical_flip = True,
                width_shift_range=0.2,
                fill_mode='constant',
                cval=0,
                height_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale =1./255)
    
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size = (256,256),
    batch_size = 4,
    class_mode = 'categorical',
    subset = 'training'
)

validation_generator = train_datagen.flow_from_directory(
    'train',
    target_size = (256,256),
    batch_size = 4,
    class_mode = 'categorical',
    subset = 'validation'
)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size = (256,256),
    batch_size = 1,
    class_mode = 'categorical',
)
def res_block(X , filter , stage ):
    #Convolutional Block 
    X_copy = X
    f1 , f2 , f3 = filter 
    
    #Main Path 
    X = Conv2D(f1 , (1,1) , strides = (1,1) , name = 'res_'+str(stage) + '_conv_a'  , kernel_initializer  = glorot_uniform(seed = 0 ))(X)
    X = MaxPool2D((2,2))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
    X = Activation('relu')(X) 

        
    X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)

    
    # Short Path
    X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
    X_copy = MaxPool2D((2,2))(X_copy)
    X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)

    X = Add()([X,X_copy])
    X = Activation('relu')(X)

    
    
    # Identity Block 1
    X_copy = X
    
    # Main Path
    
    X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
    X = Activation('relu')(X) 
    
    X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
    X = Activation('relu')(X) 

    
    X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

    # ADD
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    # Identity Block 2
    X_copy = X


    # Main Path
    X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
    X = Activation('relu')(X) 

    
    X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
    X = Activation('relu')(X) 

    
    X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)

    
    # ADD
    X = Add()([X,X_copy])
    X = Activation('relu')(X)

    return X
    

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
X = Dense(3, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)


model = Model( inputs= X_input, outputs = X, name = 'Resnet18')

model.summary()
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
model.fit_generator(train_generator, steps_per_epoch= train_generator.n // 4 , epochs = 50, validation_data= validation_generator, validation_steps= validation_generator.n // 4, callbacks=[ checkpointer ])
model.load_weights("weights.hdf5")
evaluate = model.evaluate_generator(test_generator, steps = test_generator.n // 1 , verbose = 2)
labels = {0: 'desert', 1: 'plant', 2: 'water'}
# load images and their predictions 

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import cv2

prediction = []
original = []
image = []
count = 0
for i in os.listdir('./test'):
    print(i)
    for item in os.listdir(os.path.join('./test', i)):
        # code to open the image
        img= PIL.Image.open(os.path.join('./test', i, item))
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
        img = img.reshape(-1, 256, 256, 3 )
        # making prediction of the model
        predict = model.predict(img)
        print(predict)
        # getting the index corresponding to the highest value in the prediction
        predict = np.argmax(predict)
        print(predict)
        # appending the predicted class to the list
        prediction.append(labels[predict])
        # appending original class to the list
        original.append(i)
        count = count + 1
##for i in range(len(original)):
##    print(prediction[i] , original[i])
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