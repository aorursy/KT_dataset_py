# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm    # To read in images in batches and see progress
import pathlib
import scipy
import subprocess
import gc   # Garbage collector module for memory management

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import cv2    #OpenCV for image manipulation

from tensorflow import keras  #We need keras library
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator  #Used for Data augmentation
from keras import backend as K   #For specialized and optimized tensor manipulation

from sklearn.utils import shuffle
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split  # For the creation of training and validation sets

#from six import string_types
#from IPython.display import display
#from keras.preprocessing import image as image_utils
#from keras import applications
!ls /kaggle/input/
!ls /kaggle/input/planets-dataset/
# Using DataFrame to check the shape of the training set, and their tags (labels that may be assigned) for each image
train_df = pd.read_csv('../input/planets-dataset/planet/planet/train_classes.csv')
train_df.columns = ["image_name", "tags"]
train_df

# We can see that there are indeed 40,479 training images mapped to tags.
# The second column of the sample.csv mapped each image to a tag of possible labels (separated by a space for each), that can be assigned to each image.
test_df = pd.read_csv('../input/planets-dataset/planet/planet/sample_submission.csv')
test_df
# TAG SPLITTING: Creating a list of all known tags to be assigned to the images by looping through each row in 
# the “tags” column of the train set, splitting the tags by space, and storing them in a set
label_list = []
for tag_str in train_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)

            
# Display label list and number of labels in the dataset
print(f'The number of data samples is {len(train_df)}. And there are {len(label_list)} unique possible classes.', '\n' 
      f'The Label list includes {label_list}')
# Creating a dictionary to map tags to integer so we encode and use them for modeling
# Assign a unique and consistent integer to each tag to be used to develop a target vector for each image with a One-hot encoding.
flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in train_df['tags'].values])))


# Creating a label map
label_map = {l: i for i, l in enumerate(labels)}

print(f'label_map = {label_map},\n length = {len(label_map)}')
# Creating a target vector by applying one hot encoding to the unique labels --- e.g [0 0 0 1 0 0 0 0 0 0] for "bare_ground" tag.
train_tag_data = train_df.copy()
for label in label_list:
    train_tag_data[label] = train_tag_data['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

# Display head
train_tag_data.head()

# Printing decreasing frequenciy of instances for each category
category_counts = {}

for column in train_tag_data.columns[2:]:
     category_counts[column] = train_tag_data[column].value_counts()[1]

for w in sorted(category_counts, key=category_counts.get, reverse=True):
    print(category_counts[w] , w )
# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([labels.split(" ") for labels in train_tag_data['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))

# Plotting a Histogram of label instances
tag_labels = pd.Series(labels_list).value_counts() # To sort them by count
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tag_labels, y=tag_labels.index, orient='h')
# function for cooocurence matrix plotting
def make_cooccurence_matrix(labels):
    numeric_data = train_tag_data[labels]; 
    c_matrix = numeric_data.T.dot(numeric_data)
    sns.heatmap(c_matrix)
    return c_matrix
    
# Compute the co-ocurrence matrix
make_cooccurence_matrix(label_list)
# plot land-use element classes cooccurence matrix
land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']
make_cooccurence_matrix(land_labels)
# Loading and visualizing one image in each category (or label) of the  training dataset using matplotlib
images = [train_df[train_df['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg' 
                for i, label in enumerate(labels_set)]

plt.rc('axes', grid=False)
_, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

# /kaggle/input/planets-dataset/planet/planet/train_classes.csv
# /kaggle/input/planets-dataset/planet/planet/train-jpg

for i, (image_name, label) in enumerate(zip(images, labels_set)):
    img = mpimg.imread('../input/planets-dataset/planet/planet/train-jpg' + '/' + image_name)
    axs[i].imshow(img)
    axs[i].set_title('{} - {}'.format(image_name, label))
# Determining if the length of the train and test dataset csv file equals the actual number of images in the folder

# Assign train and the two test dataset paths
# train path
train_img_dir = pathlib.Path('../input/planets-dataset/planet/planet/train-jpg')
train_img_path = sorted(list(train_img_dir.glob('*.jpg')))


# Let's read in the test image dataset and merge the test_additional jpg file to give an output of 61191 rows
# test path
test_img_dir = pathlib.Path('../input/planets-dataset/planet/planet/test-jpg')
test_img_path = sorted(list(test_img_dir.glob('*.jpg')))

# additional test path
test_add_img_dir = pathlib.Path('../input/planets-dataset/test-jpg-additional')
test_add_img_path = sorted(list(test_add_img_dir.glob('*/*.jpg')))

# Length Confirmation
assert len(train_img_path) == len(train_df)
print(len(test_img_path)+len(test_add_img_path))
# define input size. Data Length Check (or checking smapple size)
input_size = 64
x_train = []
y_train = []

for f, tags in tqdm(train_df.values, miniters=1000):
    img = cv2.imread('../input/planets-dataset/planet/planet/train-jpg/{}.jpg'.format(f))
    img = cv2.resize(img, (input_size, input_size))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(img)
    y_train.append(targets)
        
x_train = np.array(x_train, np.float32)
y_train = np.array(y_train, np.uint8)

print(x_train.shape)
print(y_train.shape)
# creating x_test
x_test = []

test_jpg_dir = '../input/planets-dataset/planet/planet/test-jpg'
test_image_names = os.listdir(test_jpg_dir)

n_test = len(test_image_names)
test_classes = test_df.iloc[:n_test, :]
add_classes = test_df.iloc[n_test:, :]


test_jpg_add_dir = '../input/planets-dataset/test-jpg-additional/test-jpg-additional'
test_add_image_names = os.listdir(test_jpg_add_dir)

for img_name, _ in tqdm(test_classes.values, miniters=1000):
    img = cv2.imread(test_jpg_dir + '/{}.jpg'.format(img_name))
    x_test.append(cv2.resize(img, (64, 64)))
    
for img_name, _ in tqdm(add_classes.values, miniters=1000):
    img = cv2.imread(test_jpg_add_dir + '/{}.jpg'.format(img_name))
    x_test.append(cv2.resize(img, (64, 64)))

x_test = np.array(x_test, np.float32)
print(x_test.shape)
gc.collect()
# split the train data into train and validation data sets
X_train = x_train[ :35000]
Y_train = y_train[ :35000]

X_valid = x_train[35000: ]
Y_valid = y_train[35000: ]
# specify sizes (batch and model input) and number of input channels
input_size = 64
input_channels = 3
batch_size = 64
model = Sequential()

# Input layer
model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))

# CCM_1
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#CCM_2
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
#CCM_3
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
#CCM_4
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Create a feature vector from the CCM_4 final layer
model.add(Flatten())

# Fully Connected (FC) Layer
model.add(Dense(512, activation='relu'))
model .add(BatchNormalization())
model.add(Dropout(0.5))

# Output layer
model.add(Dense(17, activation='sigmoid'))
import tensorflow.keras as keras
# Loading the pre-trained VGG16 architecture module
from tensorflow.keras.applications.vgg16 import VGG16



# Extract the pre - trained architecture
base_model = VGG16(input_shape =(input_size,input_size,3),include_top =False,weights ='imagenet')
base_model.summary()

# Get the output of the base_model formed above
x = base_model.output
# Flatten to obtain a feature vector
x = Flatten()(x)
# Connect the feature vector to to the fully connected (FC) layer
x = Dense (512 , activation ='relu')(x)
# Form the output label predictions
predictions = Dense (17 , activation ='sigmoid')(x)
model = Model(inputs= base_model.input,outputs = predictions)
gc.collect()
# Implementing ImageDataGenerator for data augmentation. This is an important technique which reduces 
# overfitting as it generates extra images by flipping, cropping, zooming e,t.c the images. This makes 
# the model have more images to learn from.

datagen = ImageDataGenerator ( horizontal_flip =True ,
vertical_flip =True ,
zoom_range =0.2,
rotation_range =90 ,
fill_mode ='reflect')
# Defining other parameters
epochs=20 # An epoch is one complete pass through the training data, Here, epoch is set equals 20

optimizer = keras.optimizers.Adam(learning_rate=0.0001) # Defining our Adam optimizer and learning rate
# Define the fbeta metric
def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2
 
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
 
    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)
 
    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
 
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
 
    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=[fbeta])


callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
                ModelCheckpoint(filepath='weights/best_weights',
                                 save_best_only=True,
                                 save_weights_only=True)]

gc.collect()
# The code below fits the model while generating extra images with the Imagedatagenerator, and then fit them. 
model.fit_generator(datagen.flow(X_train,
Y_train,
batch_size =24),
steps_per_epoch =len(X_train)/32 ,
validation_data = datagen.flow ( X_valid,
Y_valid,
batch_size =24),
validation_steps =len(X_valid)/32 ,
epochs =epochs ,
callbacks = callbacks ,
verbose =1)
gc.collect()
# Prediction with the trained model using the test data
test_1 =[]
test_1.append (model.predict (x_test , batch_size = 128 , verbose =2) ) 

# After prediction, we compile the results in a pandas dataframe form
result = np.array (test_1[0])
for i in range (1,len(test_1) ):
 result += np. array (test_1)
result = pd.DataFrame (result,columns = labels )
result
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.loc[[i]]
    a = a.apply(lambda x: x > 0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))
    
# The sample submission csv format
test_df['tags'] = preds
test_df.to_csv('amazon_submission11.csv', index=False)