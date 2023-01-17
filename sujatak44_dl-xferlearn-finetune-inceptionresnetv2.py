# Libraries
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm
from tqdm import tqdm_notebook


# Input shape and Batch size for InceptionResNetV2
INPUT_SHAPE = (299,299,3)
BATCH_SIZE = 10
# Name Label Dictionary
name_label_dict = {
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles"   ,
5:  "Nuclear bodies"   ,
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus"   ,
8:  "Peroxisomes"   ,
9:  "Endosomes"   ,
10:  "Lysosomes"   ,
11:  "Intermediate filaments",   
12:  "Actin filaments"   ,
13:  "Focal adhesion sites",   
14:  "Microtubules"   ,
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle"   ,
18:  "Microtubule organizing center" ,  
19:  "Centrosome"   ,
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions"  , 
23:  "Mitochondria"   ,
24:  "Aggresome"   ,
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}
#Load Train Dataset 
path_to_train = '/kaggle/input/train/'
data = pd.read_csv('/kaggle/input/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)
# Split Training Data set (80:20)
from sklearn.model_selection import train_test_split
train_ids, test_ids, train_targets, test_target = train_test_split(
    data['Id'], data['Target'], test_size=0.2, random_state=42)
# Create train dataset
def create_train(dataset_info, batch_size, shape, augument=True):
    assert shape[2] == 3
    while True:
        random_indexes = np.random.choice(len(dataset_info), batch_size)
        batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
        batch_labels = np.zeros((batch_size, 28))
        for i, idx in enumerate(random_indexes):
            image = load_image(
                dataset_info[idx]['path'], shape)   
            if augument:
                image = augment(image)
            batch_images[i] = image
            batch_labels[i][dataset_info[idx]['labels']] = 1
        yield batch_images, batch_labels
            
 # Load all the images    
def load_image(path, shape):
    R = np.array(Image.open(path+'_red.png'))
    G = np.array(Image.open(path+'_green.png'))
    B = np.array(Image.open(path+'_blue.png'))
    Y = np.array(Image.open(path+'_yellow.png'))

    image = np.stack((
        R/2 + Y/2, 
        G/2 + Y/2, 
        B),-1)

    image = cv2.resize(image, (shape[0], shape[1]))
    image = np.divide(image, 255)
    return image  
                
# augmentation            
def augment(image):
    augment_img = iaa.Sequential([
        iaa.OneOf([
            iaa.Affine(rotate=0),
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ])], random_order=True)

    image_aug = augment_img.augment_image(image)
    return image_aug
# create train datagen with augmentation
train_datagen = create_train(train_dataset_info, 5, INPUT_SHAPE, augument=True)
# Show 5 images with title, just for the refernce
images, labels = next(train_datagen)
fig, (m_axs) = plt.subplots(1, 5, figsize = (25, 10))
for i, c_ax in enumerate(m_axs.flatten()):
    c_ax.imshow(((images[i]-images.min())/(images.max()-images.min()))[:, ::1])
    c_title = '\n'.join([name_label_dict[j] for j, v in enumerate(labels[i]) if v>0.5])
    c_ax.set_title(c_title)
    c_ax.axis('off')
# Histroy Plot with loss and accuracy
def show_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    
# Load Keras Library
import tensorflow as tf
import keras
from keras import models
from keras import layers

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.models import Model
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
#Experiment 1: Freezing all layers - Same as Transfer Learning
#Load the InceptionResNetV2 model

base_model = InceptionResNetV2(
    include_top=False, 
    weights='imagenet', 
    input_shape=INPUT_SHAPE)    


# Freeze all the layers
for layer in base_model.layers[:]:
    layer.trainable = False

# Create the model
model = models.Sequential()

# Add the InceptionResNetV2 convolutional base model
model.add(base_model)

# Add new layers
model.add(Conv2D(128, kernel_size=(1,1), activation='relu'))
model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(28, activation='sigmoid'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()
#Experiment 1 : Train the model - No Data augmentation

train_generator = create_train(train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=False)
validation_generator = create_train(train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)

checkpointer = ModelCheckpoint(
    '/kaggle/working/InceptionResNetV2.model',
    verbose=2, save_best_only=True)

model.compile(
    loss='binary_crossentropy',  
    optimizer=Adam(1e-3),
    metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=20, 
    verbose=1,
    callbacks=[checkpointer])

show_history(history)
#Experiment 2 : Train Last 4 layers without data augmentation

base_model = InceptionResNetV2(
    include_top=False, 
    weights='imagenet', 
    input_shape=INPUT_SHAPE)    

# Freeze layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

    # Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(base_model)

# Add new layers
model.add(Conv2D(128, kernel_size=(1,1), activation='relu'))
model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(28, activation='sigmoid'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()
        
#Experiment 2 : Train the model - no augmentation

train_generator = create_train(train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=False)
validation_generator = create_train(train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)


model.compile(
    loss='binary_crossentropy',  
    optimizer=Adam(1e-4),
    metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=20, 
    verbose=1,
    callbacks=[checkpointer])
show_history(history)
# Experiment 3 : Train last 4 layers with data augmentation

base_model = InceptionResNetV2(
    include_top=False, 
    weights='imagenet', 
    input_shape=INPUT_SHAPE)    

# Freeze all the layers
for layer in base_model.layers[:-4]:
    layer.trainable = False
    # Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(base_model)

# Add new layers
model.add(Conv2D(128, kernel_size=(1,1), activation='relu'))
model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(28, activation='sigmoid'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()
#Experiment 3 - Train the model - with augmentation

checkpointer = ModelCheckpoint(
    '/kaggle/working/InceptionResNetV2.model',
    verbose=2, save_best_only=True)

train_generator = create_train(train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=True)
validation_generator = create_train(train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)

model.compile(
    loss='binary_crossentropy',  
    optimizer=Adam(1e-4),
    metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=20, 
    verbose=1,
    callbacks=[checkpointer])
show_history(history)
# Prediction test Dir
test_csv = pd.read_csv("../input/sample_submission.csv")
test_csv.head()

# Testing images path
TEST_PATH = "../input/test/"
ids_test = test_csv["Id"]
predicted = []
for name in tqdm(test_csv['Id']):
    path = os.path.join('../input/test/', name)
    image = load_image(path, INPUT_SHAPE)
    score_predict = model.predict(image[np.newaxis])[0]
    label_predict = np.arange(28)[score_predict>=0.2]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)
test_csv['Predicted'] = predicted
# Prepare submission file
test_csv['Predicted'].value_counts()[:20]
out_df = test_csv[['Id', 'Predicted']]
out_df = out_df[out_df.Predicted != '']
test_csv.to_csv('sample_submission.csv', index=False)

path_to_test = '/kaggle/input/test/'
#out_df['Predicted']=out_df['Predicted'].astype(str)
test_dataset_info = []    
for name, labels in zip(out_df['Id'], out_df['Predicted'].str.split(' ')):
    test_dataset_info.append({
        'path':os.path.join(path_to_test, name),
        'labels':np.array([int(label) for label in labels])})
test_dataset_info = np.array(test_dataset_info)
# create test datagen with augmentation
test_datagen = create_train(test_dataset_info, 5, INPUT_SHAPE, augument=True)

# Show 5 predicted images with title and prediction
images, labels = next(test_datagen)
predict_test = model.predict(images)
fig, (m_axs) = plt.subplots(1, 5, figsize = (25, 10))
for i, c_ax in enumerate(m_axs.flatten()):
    c_ax.imshow(((images[i]-images.min())/(images.max()-images.min()))[:, ::1])
    c_title = '\n'.join(['{}: Pred: {:2.1f}%'.format(name_label_dict[j], 100*predict_test[i, j]) 
                         for j, v in enumerate(labels[i]) if v>0.5])
   
    c_ax.set_title(c_title)
    c_ax.axis('off')