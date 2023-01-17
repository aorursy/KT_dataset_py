import keras

from keras import optimizers

from keras.preprocessing import image

from keras.engine import Layer

from keras.layers import Conv2D, Conv3D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate

from keras.layers import Activation, Dense, Dropout, Flatten

from keras.layers.normalization import BatchNormalization

from keras.callbacks import TensorBoard

from keras.models import Sequential, Model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb

from skimage.transform import resize, rescale

from skimage.io import imsave, imread

from time import time

import numpy as np

import os

import random

import tensorflow as tf

from PIL import Image, ImageFile

import matplotlib.pyplot as plt

import glob
train_dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'
def generate_rand_filename(samplesize, a, b):

  filenames = []

  for _ in range(samplesize):

    randint = str(random.randint(a,b))

    filestring = '/'+'0'*(6-len(randint))+randint+'.jpg'

    filenames.append(filestring)

  return filenames
## Visualize raw data

files = []

filepaths = []

num_examples = 4

count = 1



# for filename in glob.iglob(train_dir+'/*.jpg'):

filenames = generate_rand_filename(4, 104246, 104746)

for filename in filenames:

  files.append(filename)

  filepaths.append(train_dir+filename)

  if count > num_examples:

    break

  count += 1



files.sort()

filepaths.sort()



fig = plt.figure(figsize=(20,10))

for i in range(num_examples):

  fig.add_subplot(1, num_examples, i+1)

  plt.imshow(plt.imread(filepaths[i]))

  plt.title(files[i])
## constants

num_train_samples = 9000

num_val_samples = 1000

img_height = 224

img_width = 224

batch_size = 128

epochs = 30
class colorizationGenerator():

  

  def __init__(self, path, batch_size, filerange):

    self.path = path

    self.batch_size = batch_size

    self.filerange = filerange



  def get_input(self, path, filename):

    img = imread(path+filename)

    img = resize(img,(224,224,3))

    labImage = rgb2lab(img)

    return(labImage)



  def get_output(self, labImage):

    abch = labImage[:,:,1:]/128.0

    return np.array(abch)



  def preprocess_input(self, labImage):

    lch = labImage[:,:,0]/100.0

    l3ch = gray2rgb(lch)

    return np.array(l3ch)



  def random_flip(self, currinput, curroutput):

    if random.uniform(0, 1) > 0.5:

      currinput = currinput[:,::-1,:]

      curroutput = curroutput[:,::-1,:]

    return currinput, curroutput



  def image_generator(self):

      while True:

            # Select files (paths/indices) for the batch

            files = generate_rand_filename(self.batch_size, self.filerange[0], self.filerange[1])

            batch_paths  = np.random.choice(a    = files, 

                                            size = self.batch_size)

            batch_input  = []

            batch_output = [] 

            

            # Read in each input, perform preprocessing and get labels

            for input_path in batch_paths:

                currinput = self.get_input(self.path, input_path)

                curroutput = self.get_output(currinput)

                currinput = self.preprocess_input(currinput)



                currinput, curroutput = self.random_flip(currinput, curroutput)

                batch_input += [currinput]

                batch_output += [curroutput]

                

            # Return a tuple of (input, output) to feed the network

            batch_x = np.array(batch_input)

            batch_y = np.array(batch_output)

          

            yield(batch_x, batch_y)
## The model

modelInput = Input(shape=(img_height, img_width, 3))

vggModel = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=modelInput)



output = vggModel.layers[-1].output

vgg16TruncModel = Model(vggModel.input, output)

for layer in vgg16TruncModel.layers:

  layer.trainable=False



decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(vgg16TruncModel.output)

# decoder_output = BatchNormalization()(decoder_output)

decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)

# decoder_output = BatchNormalization()(decoder_output)

decoder_output = UpSampling2D((2, 2))(decoder_output)

decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)

# decoder_output = BatchNormalization()(decoder_output)

decoder_output = UpSampling2D((2, 2))(decoder_output)

decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)

# decoder_output = BatchNormalization()(decoder_output)

decoder_output = UpSampling2D((2, 2))(decoder_output)

decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)

# decoder_output = BatchNormalization()(decoder_output)

decoder_output = UpSampling2D((2, 2))(decoder_output)

decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)

decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=modelInput, outputs=decoder_output)
model.compile(optimizer='adam', loss='mse' , metrics=['accuracy'])



# model.compile(loss='mse',

#               optimizer=optimizers.RMSprop(lr=1e-4),

#               metrics=['accuracy'])
# dataGenerator = colorizationGenerator(train_dir, os.listdir(train_dir), batch_size)

# valGenerator = colorizationGenerator(val_dir, os.listdir(val_dir), num_val_samples)

dataGenerator = colorizationGenerator(train_dir, batch_size, [1, 1 + num_train_samples])

valGenerator = colorizationGenerator(train_dir, batch_size, [200000, 200000 + num_val_samples])



from keras.callbacks import ModelCheckpoint



checkpoint_callback = ModelCheckpoint('best_model_wval.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
test_img = imread(train_dir+'/104247.jpg')

test_img = resize(test_img,(224,224,3))

lab = rgb2lab(test_img)

print(lab.shape)

l = lab[:,:,0]

a = lab[:,:,1]

b = lab[:,:,2]

print(np.max(l))

print(np.max(a))

print(np.max(b))

plt.imshow(test_img)
images = next(dataGenerator.image_generator())

print(len(images))

print(images[1][0].shape)

print(np.max(images[0][0][0]))

print(np.max(images[1][0][:,:,0]))

print(np.max(images[1][0][:,:,1]))

cur = np.zeros((224, 224, 3))

cur[:,:,0] = images[0][0][:,:,0]*100.0

cur[:,:,1:] = images[1][0]*128.0

output = lab2rgb(cur)

plt.imshow(output)

print(np.max(cur[:,:,0]))

print(np.max(cur[:,:,1]))

print(np.max(cur[:,:,2]))
history = model.fit_generator(

    dataGenerator.image_generator(),

    steps_per_epoch=num_train_samples // batch_size,

    validation_data=valGenerator.image_generator(),

    validation_steps=num_val_samples // batch_size,

    epochs=epochs,

    callbacks=[checkpoint_callback]

)
model.save('last_model.h5')
acc = history.history['accuracy']

mse = history.history['loss']



epochs_range = range(len(acc))



plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, mse, label='Training MSE')

plt.plot(history.history['val_loss'], label='Validation MSE')

plt.legend(loc='upper right')

plt.title('Training MSE')



plt.show()
testfiles = []

testfilepaths = []

val_index = 200000

num_examples = 10

fig = plt.figure(figsize=(20,10))

for i in range(val_index, val_index + num_examples):

  fig.add_subplot(1, num_examples, i+1-val_index)

  filename = str(i) + '.jpg' 

  testfiles.append(filename)

  testfilepaths.append(os.path.join(train_dir,filename))

  plt.imshow(plt.imread(os.path.join(train_dir,filename)))

  plt.title(str(i) + '.jpg')
from keras.models import load_model

bestModel = load_model('last_model.h5')
inputs = []

outputs = []

for idx, file in enumerate(testfilepaths):

    test = imread(file)

    test = resize(test, (224,224,3), anti_aliasing=True)

    lab = rgb2lab(test)

    l = lab[:,:,0]

    L = gray2rgb(l/100.0)

    inputs.append(L)

    L = L.reshape((1,224,224,3))

    ab = bestModel.predict(L)

    ab = ab*128

    cur = np.zeros((224, 224, 3))

    cur[:,:,0] = l

    cur[:,:,1:] = ab

    outputs.append(cur)
num_examples = 10

fig = plt.figure(figsize=(20,10))

for i in range(num_examples):

  fig.add_subplot(1, num_examples, i+1)

  plt.imshow(inputs[i])
num_examples = 10

fig = plt.figure(figsize=(20,10))

for i in range(num_examples):

  fig.add_subplot(1, num_examples, i+1)

  plt.imshow(lab2rgb(outputs[i]))
plt.imshow(outputs[0][:,:,1])
plt.imshow(outputs[0][:,:,2])
bestModel = load_model('best_model_wval.h5')

inputs = []

outputs = []

for idx, file in enumerate(testfilepaths):

    test = imread(file)

    test = resize(test, (224,224,3), anti_aliasing=True)

    lab = rgb2lab(test)

    l = lab[:,:,0]

    L = gray2rgb(l/100.0)

    inputs.append(L)

    L = L.reshape((1,224,224,3))

    ab = bestModel.predict(L)

    ab = ab*128

    cur = np.zeros((224, 224, 3))

    cur[:,:,0] = l

    cur[:,:,1:] = ab

    outputs.append(cur)
num_examples = 10

fig = plt.figure(figsize=(20,10))

for i in range(num_examples):

  fig.add_subplot(1, num_examples, i+1)

  plt.imshow(inputs[i])
num_examples = 10

fig = plt.figure(figsize=(20,10))

for i in range(num_examples):

  fig.add_subplot(1, num_examples, i+1)

  plt.imshow(lab2rgb(outputs[i]))