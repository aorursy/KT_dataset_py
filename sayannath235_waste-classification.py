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
train_dir = '../input/waste-classification-data/DATASET/TRAIN'
test_dir = '../input/waste-classification-data/DATASET/TEST'
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as imread

import tensorflow
print(tensorflow.__version__)

import keras
print(keras.__version__)
from IPython.display import clear_output
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Sequential
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
import os
print(os.listdir(train_dir))
print(os.listdir(test_dir))
# Training Data with R/O pictures
train_R_dir = os.path.join(train_dir, 'R')
train_O_dir = os.path.join(train_dir, 'O')

#Validation Data with R/O pictures
validation_R_dir = os.path.join(test_dir, 'R')
validation_O_dir = os.path.join(test_dir, 'O')
train_R_frames = os.listdir(train_R_dir)
train_O_frames = os.listdir(train_O_dir)

#Viewing the first 10 filenames
print(train_R_frames[:10])
print(train_O_frames[:10])
print("Number of training R images: ", len(os.listdir(train_R_dir)))
print("Number of training O images: ", len(os.listdir(train_O_dir)))

print("Number of validation R images: ", len(os.listdir(validation_R_dir)))
print("Number of validation O images: ", len(os.listdir(validation_O_dir)))
#To get the file name of the training image
print(os.listdir(train_R_dir)[0])
print(os.listdir(train_O_dir)[0])
#Setting the filepath of the images of both R and O
R_image = os.path.join(train_R_dir, 'R_2234.jpg')
O_image = os.path.join(train_O_dir, 'O_9571.jpg')
import matplotlib
R_img = matplotlib.image.imread(R_image)
O_img = matplotlib.image.imread(O_image)
print("Image shape of training images of R: ", R_img.shape)
print("Image shape of training images of O: ", O_img.shape)
dim1 = []
dim2 = []
for image_filename in os.listdir(test_dir+'/R'):
    
    img = matplotlib.image.imread(test_dir+'/R'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
print(np.mean(d1))
print(np.mean(d2))
image_size = (225, 225)
sns.jointplot(dim1,dim2)
%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_R_pix = [os.path.join(train_R_dir, fname) 
                for fname in train_R_frames[ pic_index-8:pic_index] 
               ]

next_O_pix = [os.path.join(train_O_dir, fname) 
                for fname in train_O_frames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_R_pix+next_O_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

datagen = ImageDataGenerator(
    rotation_range=20,       # Rotate the image 20 degrees
    width_shift_range=0.10,  # Shift the image width by a max of 5%
    height_shift_range=0.10, # Shift the image height by a max of 5%
    rescale=1./255,          # Rescale the image by normalzing it.
    shear_range=0.1,         # Shear means cutting away part of the image (max 10%)
    zoom_range=0.1,          # Zoom in by 10% max
    horizontal_flip=True,    # Allow horizontal flipping
    fill_mode='nearest'      # Fill in missing pixels with the nearest filled value
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=128,
    color_mode='rgb',
    class_mode='binary'
)

validation_gen = datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=128,
    color_mode='rgb',
    class_mode='binary'
)
print (train_gen.class_indices)

labels = '\n'.join(sorted(train_gen.class_indices.keys()))

with open('labels.txt', 'w') as f:
  f.write(labels)
class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []
        

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('Log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="acc")
        ax2.plot(self.x, self.val_acc, label="val_acc")
        ax2.legend()
        
        plt.show()
        
        
plot = PlotLearning()
