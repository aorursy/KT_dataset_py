# Copy python scripts from my library my_python, which contains the clr_callback script
# to the working directory.
! cp ../input/my-python/* ../working/
# Generic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import gc
from tqdm import tqdm_notebook
%matplotlib inline
sns.set_style('whitegrid')

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Keras imports
from keras.applications import VGG16, ResNet50, ResNet50V2, InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Other imports
from clr_callback import *   
import imgaug as ia
from imgaug import augmenters as iaa
from IPython.display import FileLink, FileLinks
# Constants
DATA_DIR = '../input/dog-breed-identification/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
BATCH_SIZE = 128
INPUT_SIZE = 299
NUM_CLASSES = 120
SEED = 42
# Let's check what's in the data directory
! ls $DATA_DIR
# Read the train data set, which has the ids of the images and their labels (breeds)
# (adding the extension .jpg to the id becomes the file name of the image) 
train = pd.read_csv(DATA_DIR + 'labels.csv')
train.head()
breeds_20 = train.breed.value_counts().head(NUM_CLASSES).index
train = train[train.breed.isin(breeds_20)]
# The submission file contains one column for the image id, and then one column 
# each breed in alphabetical order, with the probability of the dog in the image beeing of that breed
submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')
submission.head()
# Get the breeds to pass them to the generators for creating the same labels for train set and validation set
#breed_labels = list(submission.columns[1:].values)
breed_labels = list(breeds_20.values)
# Now we create our model

def create_model(lr=0.0001, dropout=None):
    
    model = Sequential()
    
    #base = InceptionV3(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    #base = Xception(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    #base = VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    #base = ResNet50(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    #base.layers.pop()   # Remove the last layer (softmax classifier)

    for layer in base.layers:
        layer.trainable = False
    
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    adam = Adam(lr=lr)
    sgd = SGD(lr=0.1, momentum=0.95, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'],  )

    return model
# We split images into training (80%) and validation (20%)
train_set, val_set = train_test_split(train, test_size=0.20, stratify=train['breed'], random_state=SEED)
# Create the generators and data augmenters

def keypoints(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images

def rescale_imgs(images, random_state, parents, hooks):
    for img in images:
        img = img / 255.
    return images 
    
augs = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Sometimes(0.2, iaa.Affine(rotate=(-20,20), mode='edge')),  
    iaa.SomeOf((0,4), [
        iaa.AdditiveGaussianNoise(scale=0.01*255),        
        iaa.Sharpen(alpha=(0.0,0.3)),
        iaa.ContrastNormalization((0.8,1.2)),
        iaa.AverageBlur(k=(2,11)),
        iaa.Multiply((0.8,1.2)),
        iaa.Add((-20,20), per_channel=0.5),
        iaa.Grayscale(alpha=(0.0,1.0))
    ])
    #,    iaa.Lambda(rescale_imgs, keypoints)
]) 

train_datagen = ImageDataGenerator(preprocessing_function = augs.augment_image)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_set,
                                                directory=TRAIN_DIR,
                                                x_col='id',
                                                y_col='breed',
                                                class_mode='categorical',
                                                classes=breed_labels,
                                                has_ext=False,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                seed=SEED,
                                                target_size=(INPUT_SIZE, INPUT_SIZE)
                                               )

valid_generator = val_datagen.flow_from_dataframe(dataframe=val_set,
                                                directory=TRAIN_DIR,
                                                x_col='id',
                                                y_col='breed',
                                                class_mode='categorical',
                                                classes=breed_labels,
                                                has_ext=False,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                seed=SEED,target_size=(INPUT_SIZE, INPUT_SIZE)
                                               )

gc.collect()

model = create_model()

train_generator.reset()
valid_generator.reset()

STEP_SIZE_TRAIN = train_set.shape[0] // BATCH_SIZE
STEP_SIZE_VALID = val_set.shape[0] // BATCH_SIZE

EPOCHS = 1
base_lr=0.0001
max_lr=1
step_size = EPOCHS * STEP_SIZE_TRAIN 
clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step_size)
model.fit_generator(train_generator, 
                              epochs=EPOCHS, 
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator,
                              validation_steps=STEP_SIZE_VALID,
                              callbacks=[clr]
                             )
# Let's plot accuracies against learning rates to select the base_lr and max_lr following the article

results = pd.DataFrame(clr.history)
lr = results['lr']
loss = results['loss']
window=1
rolling_loss = loss.rolling(window).mean().fillna(0)

fig = plt.figure(figsize=(20, 10))
ticks = np.arange(0, 0.5, 0.005)
labels = ticks
plt.xticks(ticks, ticks, rotation='vertical')
plt.tick_params(axis='x', which='minor', colors='black')
#plt.xscale('log')
every = 2
#plt.ylim(4, 5.25)
plt.xlim(0,0.5)
till=500
plt.plot(lr[::every], loss[::every])
plt.show()
# Create the model in a different cell, just in case we want to train it several times
model = create_model()
model.summary()
# Training

gc.collect()

EPOCHS=40
STEP_SIZE_TRAIN = train_set.shape[0] // BATCH_SIZE
STEP_SIZE_VALID = val_set.shape[0] // BATCH_SIZE

train_generator.reset()
valid_generator.reset()

clr = CyclicLR(base_lr=0.04, max_lr=0.11, step_size=2*STEP_SIZE_TRAIN, mode='triangular')
checkpoint = ModelCheckpoint('dog_breed_inceptionv3.hf5', monitor='val_acc', verbose=0, save_best_only=True, mode='max', save_weights_only=False)
reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr = 0.00001)
history = model.fit_generator(train_generator, 
                              epochs=EPOCHS, 
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator,
                              validation_steps=STEP_SIZE_VALID,
                              callbacks=[checkpoint, clr]
                            
                             )
# Create a link to download the model weights
FileLink('dog_breed_inceptionv3.hf5')
# Plot loss and accuracy both for train and validateion sets
def plt_history(history, metric, title, ax, val=True):
    ax.plot(history[metric])
    if val:
        ax.plot(history['val_' + metric])
    ax.grid(True)
    ax.set_title(title)
    ax.xaxis.set_ticks(range(0,EPOCHS))
    ax.xaxis.set_ticklabels([str(i) for i in range(1,EPOCHS+1)])
    

    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    
    
hist = history.history
fig, ax = plt.subplots(1,2, figsize=(18,6))
plt_history(hist, 'loss', 'LOSS', ax[0])
plt_history(hist, 'acc', 'ACCURACY', ax[1])
plt.savefig('history.png')
# Load the best model

model.load_weights('dog_breed_inceptionv3.hf5')

# Create a generator from the submission dataframe to leverage model.predict_generator to
# make the predictions

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(dataframe=submission,
                                                directory=TEST_DIR,
                                                x_col='id',
                                                class_mode=None,
                                                has_ext=False,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                seed=SEED,
                                                target_size=(INPUT_SIZE, INPUT_SIZE)
                                               )

predictions = model.predict_generator(test_generator, verbose=1)
# Substitute the dummy predictions in submmission by the model predictions
submission.loc[:,1:] = predictions
submission.head()
# Save the submission to a file and create a link to download it (without the need of commiting)
submission.to_csv('submission.csv', index=False)
FileLink('submission.csv')