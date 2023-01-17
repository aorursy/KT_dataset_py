!pip install -q kaggle
# Put kaggle.json in the working directory
# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json
# download dataset
# Copy the stackoverflow data set locally.
!kaggle datasets download jbeltranleon/xrays-chest-299-small
shutil.rmtree('299_small')

# Unzip the data
!unzip -qq -n xrays-chest-299-small.zip
# Switch directory and show its content
!cd 299_small && ls
import shutil

shutil.rmtree('input_299_small')
#shutil.rmtree('sample_data')
import os

base_dir = '299_small'

# Directory to our training data
train_folder = os.path.join(base_dir, 'train')

# Directory to our validation data
val_folder = os.path.join(base_dir, 'val')

# Directory to our validation data
test_folder = os.path.join(base_dir, 'test')

# List folders and number of files
print("Directory, Number of files")
for root, subdirs, files in os.walk(base_dir):
    print(root, len(files))
#labels of 9 classes
label = ["atelectasis","cardiomegaly","consolidation","effusion","infiltration","mass","no_finding","nodule","pneumothorax"]
# labels for each of the 9 sub-models
LABELS = [['effusion','infiltration','mass','no_finding'],
          ['effusion','infiltration','no_finding','nodule'],
          ['effusion','infiltration','no_finding','pneumothorax'],
          ['atelectasis','effusion','infiltration','no_finding'],
          ['atelectasis','infiltration','mass','no_finding'],
          ['atelectasis','infiltration','no_finding','nodule'],
          ['atelectasis','infiltration','no_finding','pneumothorax'],
          ['consolidation','infiltration','no_finding'],
          ['cardiomegaly','infiltration','no_finding']]
# total sizes of each of the 9 classes
size = [2697,699,838,2531,6109,1368,6400,1731,1404]
# sizes for each of the 9 sub-datasets
SIZES = [[632,679,684,711],
         [633,679,711,865],
         [633,679,712,702],
         [675,633,679,711],
         [674,679,684,711],
         [674,679,711,866],
         [674,679,711,702],
         [838,678,711],
         [699,678,711]]
#for i in range(9):
  #shutil.rmtree('299_small/train' + str(i))

for i in range(9):
  os.mkdir('299_small/train' + str(i))
  for j in range(len(LABELS[i])):
    os.mkdir('299_small/train' + str(i) + "/" + LABELS[i][j])
import numpy as np

for i in range(9):
  for j in range(len(SIZES[i])):
    print(LABELS[i][j])
    path, dirs, files = next(os.walk('299_small/train/' + LABELS[i][j]))
    print(len(files))
    print(SIZES[i][j])
    for k in range(SIZES[i][j]):
      shutil.move('299_small/train/' + LABELS[i][j]+'/'+files[k], '299_small/train' + str(i) + '/' + LABELS[i][j])
for i in range(9):
  os.mkdir('299_small/val' + str(i))
  for j in range(len(LABELS[i])):
    os.mkdir('299_small/val' + str(i) + "/" + LABELS[i][j])
import numpy as np

for i in range(9):
  for j in range(len(SIZES[i])):
    print(LABELS[i][j])
    path, dirs, files = next(os.walk('299_small/val/' + LABELS[i][j]))
    print(len(files))
    for k in range(len(files)):
      shutil.copy('299_small/val/' + LABELS[i][j]+'/'+files[k], '299_small/val' + str(i) + '/' + LABELS[i][j])
from keras.preprocessing.image import ImageDataGenerator

train_folder = '299_small/train0'
val_folder = '299_small/val0'

# Batch size
bs = 16

# All images will be resized to this value
image_size = (299, 299)

# All images will be rescaled by 1./255. We apply data augmentation here.

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 16 using train_datagen generator
print("Preparing generator for train dataset")
train_generator = train_datagen.flow_from_directory(
    directory= train_folder, # This is the source directory for training images 
    target_size=image_size, # All images will be resized to value set in image_size
    batch_size=bs,
    class_mode='categorical')

# Flow validation images in batches of 16 using val_datagen generator
print("Preparing generator for validation dataset")
val_generator = val_datagen.flow_from_directory(
    directory= val_folder, 
    target_size=image_size,
    batch_size=bs,
    class_mode='categorical')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D
from tensorflow.keras.applications.densenet import DenseNet121

# Here we specify the input shape of our data 
# This should match the size of images ('image_size') along with the number of channels (3)
input_shape = (299, 299, 3)

# Define the number of classes
num_classes = 14

input_img = Input(shape=input_shape)

model = DenseNet121(include_top=False, weights=None, input_tensor=input_img, input_shape=input_shape, pooling='avg', classes = num_classes)
temp = model.layers[-1].output

predictions = Dense(14, activation='softmax',name = 'last')(temp)

model = Model(inputs=input_img, outputs=predictions)
# load weigths from model trained on ChestXray14 datatset for transfer learning

model_path = 'brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
model.load_weights( model_path )
temp = model.layers[-2].output

num_classes = 4
predictions = Dense(num_classes, activation='softmax',name = 'last')(temp)

model = Model(inputs=input_img, outputs=predictions)

# only train last dense layer first
for layer in model.layers:
    if layer.name != 'last':
        layer.trainable = False
model.summary()
from tensorflow.keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint

Checkpointer = ModelCheckpoint('Ensemble0.hdf5', monitor='val_accuracy', save_best_only=True, verbose=1)
history = model.fit(
        train_generator, # train generator has 23777 train images
        steps_per_epoch=train_generator.samples // bs + 1,
        epochs=120,
        validation_data=val_generator, # validation generator has 5948 validation images
        validation_steps=val_generator.samples // bs + 1,
        callbacks=[Checkpointer]
)
from tensorflow.keras.models import load_model

model_path = 'Ensemble0.hdf5'
model = load_model( model_path )
model.summary()
# gradually unfreeze weights from the output to the input and train the whole model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
Checkpointer = ModelCheckpoint('Ensemble0prime.hdf5', monitor='val_accuracy', save_best_only=True, verbose=2)
from tensorflow.keras.models import load_model
model_path = 'Ensemble0prime.hdf5'
model = load_model( model_path )
for i in range (1,16):
  for j in range(1,27):
    z=i*27+j+1
    if z > 429:
      break
    model.layers[-z].trainable = True
  model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])
  history = model.fit(
        train_generator, # train generator has 23777 train images
        steps_per_epoch=train_generator.samples // bs + 1,
        epochs=20,
        verbose = 2,
        validation_data=val_generator, # validation generator has 5948 validation images
        validation_steps=val_generator.samples // bs + 1,
        callbacks=[Checkpointer])
  
  model = load_model( model_path )
  print(i)
  