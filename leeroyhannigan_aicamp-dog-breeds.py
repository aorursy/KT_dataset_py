import pandas as pd         #read in csv data
import numpy as np 
import tqdm as tqdm         # graphical representation of progress
import cv2 as cv            # Open images
import matplotlib.pyplot as plt    # Plot graphs
import matplotlib.image as mpimg

from keras.layers import Activation, Convolution2D, Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.applications import InceptionV3, ResNet50, Xception, VGG16
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from numpy.random import seed
#make directory to put weights
from pathlib import Path
import os
# mkdir -p ~/.keras/models
cache_dir = Path.home() / '.keras'
if not cache_dir.exists():
    os.makedirs(cache_dir)
models_dir = cache_dir / 'models'
if not models_dir.exists():
    os.makedirs(models_dir)
!cp  ../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/
%matplotlib inline
seed(seed=1982)
#Create Path
PATH = '../input/dog-breed-identification/'
#Get labels and sample submissions
train_df = pd.read_csv(PATH + 'labels.csv')
test_df = pd.read_csv(PATH+'sample_submission.csv')
train_df.head()
#Declare Variables
NUM_CLASSES = 16
IMG_HEIGHT=250
IMG_WIDTH = 250
#array for resized images
images =[]
#array for labels
classes=[]
#select 16 breeds
selected_breeds = list(train_df.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
#Array of images that match our 16 breeds
sub_df_train =  train_df[train_df.breed.isin(selected_breeds)]
#Create array of breeds
targets = pd.Series(sub_df_train.breed)
targets.head(20)
#get_dummies changes label strings to numerical vale
one_hot = pd.get_dummies(targets, sparse=True)
one_hot.head()
#assign onehot numerical values to numpy array
one_hot_labels = np.asarray(one_hot)
print((one_hot_labels[:5]))
#Get Images, resize and assign to images array
for f, breeds in tqdm.tqdm(sub_df_train.values):
    img= cv.imread('../input/dog-breed-identification/train/{}.jpg'.format(f))        
    images.append(cv.resize(img, (IMG_HEIGHT,IMG_WIDTH)))
#assign images and labes to numpy arrays
classes = one_hot_labels
X = np.asarray(images, dtype=np.float32)
Y = np.asarray(classes, dtype=np.uint8)

print(X.shape)
print(Y.shape)
#train test split takes train and test data and labels. in this case 70% train, 30% test
X_train, X_val, y_train, y_val = train_test_split(X,Y, test_size=0.3, shuffle=True)
#declare model
starting_model = InceptionV3(include_top=False
                             ,weights='imagenet'
                             , input_shape=(IMG_HEIGHT,IMG_WIDTH,3),
                             classes=NUM_CLASSES)
#Add our own layers to the fully connected layer
model = starting_model.output
model = Flatten()(model)
model = Dense(1024, activation='relu')(model)
model = Dropout(0.5)(model)
model = Dense(1024, activation='relu')(model)

predictions = Dense(NUM_CLASSES, activation='softmax')(model)

final_model = Model(inputs=[starting_model.input], outputs=[predictions])
#compile with grid classifer
final_model.compile(SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
#get dataframes
train_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
valid_generator = valid_datagen.flow(X_val, y_val, batch_size = 32)

#fit model
final_model.fit_generator(train_generator, epochs=30, validation_data=valid_generator)
