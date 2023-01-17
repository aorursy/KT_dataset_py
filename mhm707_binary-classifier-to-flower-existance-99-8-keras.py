from google.colab import drive
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, ZeroPadding2D, BatchNormalization
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, image
tf.test.gpu_device_name()

original_dataset_dir = '/content/drive/My Drive/FD' 

base_dir = '/content/FDF_and_NF'
#os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)

test_dir = os.path.join(base_dir, 'test') 
#os.mkdir(test_dir)

train_flower_dir = os.path.join(train_dir, 'Flower')
#os.mkdir(train_flower_dir)

train_nonflower_dir = os.path.join(train_dir, 'NF')
#os.mkdir(train_nonflower_dir)

test_flower_dir = os.path.join(test_dir, 'Flower')
#os.mkdir(test_flower_dir)

test_nonflower_dir = os.path.join(test_dir, 'NF')
#os.mkdir(test_nonflower_dir)


fnames = ['FD.{}.jpg'.format(i) for i in range(4700)]
for fname in fnames:
  src = os.path.join(original_dataset_dir, fname)
  dst = os.path.join(train_flower_dir, fname)
  shutil.copyfile(src, dst)


fnames = ['FD.{}.jpg'.format(i) for i in range(4700, 5262)]
for fname in fnames:
  src = os.path.join(original_dataset_dir, fname)
  dst = os.path.join(test_flower_dir, fname)
  shutil.copyfile(src, dst)

  fnames = ['NFD{}.jpg'.format(i) for i in range(4700)]
for fname in fnames:
  src = os.path.join(original_dataset_dir, fname)
  dst = os.path.join(train_nonflower_dir, fname)
  shutil.copyfile(src, dst)

fnames = ['NFD{}.jpg'.format(i) for i in range(4700, 5262)]
for fname in fnames:
  src = os.path.join(original_dataset_dir, fname)
  dst = os.path.join(test_nonflower_dir, fname)
  shutil.copyfile(src, dst)


print('total training flower images:', len(os.listdir(train_flower_dir)))
print('total training non-flower images:', len(os.listdir(train_nonflower_dir)))
print('total test flower images:', len(os.listdir(test_flower_dir)))
print('total test non-flower images:', len(os.listdir(test_nonflower_dir)))

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    class_mode = 'binary',
    target_size = (200, 200),
    batch_size = 16)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    class_mode = 'binary',
    target_size = (200, 200),
    batch_size = 16)

for data_batch, labels_batch in train_generator:
  print('data batch shape:', data_batch.shape)
  print('labels batch shape:', labels_batch.shape)
  break
def f_n(input_shape):

  #Placeholding for the X_input.
  X_input = Input(input_shape)
    
  X = X_input

  X = Conv2D(32, (3, 3), activation = 'relu', input_shape=(200, 200, 3))(X)
  X = MaxPooling2D((2, 2))(X)
  X = Conv2D(32, (3, 3), activation = 'relu')(X)
  X = MaxPooling2D((2, 2))(X)
  X = Flatten()(X)
  X = Dropout(0.5)(X)

  #FC
  X = Dense(16, activation = 'relu')(X)
  #Sigmoid activation
  X = Dense(1, activation = 'sigmoid')(X)

  #Model creation
  model = Model(inputs = X_input, outputs = X, name='f_n')

  return model
F_N = f_n(input_shape = (200, 200, 3)) #Assigning the model
F_N.compile(loss = 'binary_crossentropy', 
              optimizer = 'Adam',
              metrics = ['acc']) 
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size #Determining the step size == (number of samples)/(batch size)

F_N.fit_generator(generator=train_generator,                # Model training
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs = 30)
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size #Determining the step size == (number of samples)/(batch size)

test_generator.reset()

pred = F_N.predict_generator(test_generator,        # Model Evaluation
steps=STEP_SIZE_TEST,
verbose=1)

print ("Loss = " + str(pred[0]))
print ("Test Accuracy = " + str(pred[1]))
F_N.save('F_N_99.9%,h5') #Saving the weights of the model as an h5 file.
F_N = load_model('/content/drive/My Drive/Colab Notebooks/Flower_Not/F_N_99%,h5') # Only if there is already a trained model !
F_N.summary() 
from google.colab import files        #Test your own images ! 
uploaded = files.upload()             #Upload an image from your dir.

for name, data in uploaded.items():
  with open(name, 'wb') as f:
    f.write(data)
    print ('saved file', name)
from matplotlib.pyplot import imshow
from keras.applications.imagenet_utils import preprocess_input

img_path = '/content/' + name          #Uncomment if you want to use the image uploded by the previous cell.
#img_path = '/content/' + '350' + '.jpg' #Uncomment if you want to choose the image manually.

img = image.load_img(img_path, target_size=(200, 200))

imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

if F_N.predict(x) == 0 :
  print("It contains flower !")
else :
  print("It does not contain flower")
