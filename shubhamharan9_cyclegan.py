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
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
!pip install git+https://www.github.com/keras-team/keras-contrib.git
from glob import glob
from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import Add, Conv2DTranspose, ZeroPadding2D, LeakyReLU
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from imageio import imread
from skimage.transform import resize
def residual_block(x):
  
  res = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(x)
  res = BatchNormalization(axis = 3, momentum = 0.9, epsilon = 1e-5)(res)
  res = Activation('relu')(res)
  res = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(res)
  res = BatchNormalization(axis = 3, momentum = 0.9, epsilon = 1e-5)(res)
  
  return Add()([res, x])
def save_images(originalA, generatedB, reconstructedA,
                originalB, generatedA, reconstructedB, path):
  
  """
  Save images
  """
  
  fig = plt.figure(figsize=(12,12))
  
  ax = fig.add_subplot(2, 3, 1)
  ax.imshow(originalA)
  ax.axis("off")
  ax.set_title("Original")
  
  ax = fig.add_subplot(2, 3, 2)
  ax.imshow(generatedB)
  ax.axis("off")
  ax.set_title("Generated")
  
  ax = fig.add_subplot(2, 3, 3)
  ax.imshow(reconstructedA)
  ax.axis("off")
  ax.set_title("Reconstructed")
  
  ax = fig.add_subplot(2, 3, 4)
  ax.imshow(originalB)
  ax.axis("off")
  ax.set_title("Original")
  
  ax = fig.add_subplot(2, 3, 5)
  ax.imshow(generatedA)
  ax.axis("off")
  ax.set_title("Generated")
  
  ax = fig.add_subplot(2, 3, 6)
  ax.imshow(reconstructedB)
  ax.axis("off")
  ax.set_title("Reconstructed")
  plt.show()
  
  #plt.savefig(path)
  
## Generator Network

def build_generator():

  """
  Creating a generator network with the hyperparameters defined below
  """
  
  input_shape = (128, 128, 3)
  residual_blocks = 6
  input_layer = Input(shape = input_shape)
  
  
  ## 1st Convolutional Block
  x = Conv2D(filters = 32, kernel_size = 7, strides = 1, padding = "same")(input_layer)
  x = InstanceNormalization(axis = 3)(x)
  x = Activation("relu")(x)
  
  ## 2nd Convolutional Block
  x = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = "same")(x)
  x = InstanceNormalization(axis = 3)(x)
  x = Activation("relu")(x)
  
  ## 3rd Convolutional Block
  x = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same")(x)
  x = InstanceNormalization(axis = 3)(x)
  x = Activation("relu")(x)
  
  
  ## Residual blocks
  for _ in range(residual_blocks):
    x = residual_block(x)
  
  
  ## 1st Upsampling Block
  x = Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = "same", use_bias = False)(x)
  x = InstanceNormalization(axis = 3)(x)
  x = Activation("relu")(x)
  
  ## 2nd Upsampling Block
  x = Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2, padding = "same", use_bias = False)(x)
  x = InstanceNormalization(axis = 3)(x)
  x = Activation("relu")(x)
  
  ## Last Convolutional Layer
  x = Conv2D(filters = 3, kernel_size = 7, strides = 1, padding = "same")(x)
  output = Activation("tanh")(x)
  
  
  model = Model(inputs = [input_layer], outputs = [output])
  return model
  
  

## Generator Network

def build_generator():

  """
  Creating a generator network with the hyperparameters defined below
  """
  
  input_shape = (128, 128, 3)
  residual_blocks = 6
  input_layer = Input(shape = input_shape)
  
  
  ## 1st Convolutional Block
  x = Conv2D(filters = 32, kernel_size = 7, strides = 1, padding = "same")(input_layer)
  x = InstanceNormalization(axis = 3)(x)
  x = Activation("relu")(x)
  
  ## 2nd Convolutional Block
  x = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = "same")(x)
  x = InstanceNormalization(axis = 3)(x)
  x = Activation("relu")(x)
  
  ## 3rd Convolutional Block
  x = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same")(x)
  x = InstanceNormalization(axis = 3)(x)
  x = Activation("relu")(x)
  
  
  ## Residual blocks
  for _ in range(residual_blocks):
    x = residual_block(x)
  
  
  ## 1st Upsampling Block
  x = Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = "same", use_bias = False)(x)
  x = InstanceNormalization(axis = 3)(x)
  x = Activation("relu")(x)
  
  ## 2nd Upsampling Block
  x = Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2, padding = "same", use_bias = False)(x)
  x = InstanceNormalization(axis = 3)(x)
  x = Activation("relu")(x)
  
  ## Last Convolutional Layer
  x = Conv2D(filters = 3, kernel_size = 7, strides = 1, padding = "same")(x)
  output = Activation("tanh")(x)
  
  
  model = Model(inputs = [input_layer], outputs = [output])
  return model
  
  

## Discriminator Network

def build_discriminator():

  """
  Creating a discriminator network using the hyperparameters defined below
  """
  
  input_shape = (128, 128, 3)
  hidden_layers = 3
  
  input_layer = Input(shape = input_shape)
  
  x = ZeroPadding2D(padding = (1, 1))(input_layer)
  

  ## 1st Convolutional Block
  x = Conv2D(filters = 64, kernel_size = 4, strides = 2, padding = "valid")(x)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = ZeroPadding2D(padding = (1, 1))(x)
  
  
  ## 3 Hidden Convolutional Blocks
  for i in range(1, hidden_layers + 1):
    x = Conv2D(filters = 2 * i * 64, kernel_size = 4, strides = 2, padding = "valid")(x)
    x = InstanceNormalization(axis = 3)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    
    x = ZeroPadding2D(padding = (1, 1))(x)
    
  
  ## Last Convolutional Layer
  output = Conv2D(filters = 1, kernel_size = 4, strides = 1, activation = "sigmoid")(x)
  
  
  model = Model(inputs = [input_layer], outputs = [output])
  return model
  
  

def load_images(data_dir):
  
  imagesA = glob(data_dir + '/testA/*.*')
  imagesB = glob(data_dir + '/testB/*.*')
  #imagesB.append(glob(data_dir + '/testB/*.*'))
  

  
  allImagesA = []
  allImagesB = []
  
  for index, filename in enumerate(imagesA):
    imgA = imread(filename, pilmode = "RGB")
    imgB = imread(imagesB[index], pilmode = "RGB")
    
    imgA = resize(imgA, (128, 128))
    imgB = resize(imgB, (128, 128))
    allImagesA.append(imgA)
    allImagesB.append(imgB)
    
  
  ## Normalize images
  allImagesA = np.array(allImagesA) / 127.5 - 1.
  allImagesB = np.array(allImagesB) / 127.5 - 1.
  
  return allImagesA, allImagesB
def load_test_batch(data_dir, batch_size):
  
  imagesA = glob(data_dir + '/testA/*.*')
  imagesB = glob(data_dir + '/testB/*.*')
  
  imagesA = np.random.choice(a = imagesA, size = batch_size)
  imagesB = np.random.choice(a = imagesB, size = batch_size)
  
  allA = []
  allB = []
  
  for i in range(len(imagesA)):
    ## Load and resize images
    imgA = resize(imread(imagesA[i], pilmode = 'RGB').astype(np.float32), (128, 128))
    imgB = resize(imread(imagesB[i], pilmode = 'RGB').astype(np.float32), (128, 128))
    
    allA.append(imgA)
    allB.append(imgB)
    
  return np.array(allA) / 127.5 - 1.0, np.array(allB) / 127.5 - 1.0
def load_test_batch(data_dir, batch_size):
  
  imagesA = glob(data_dir + '/testA/*.*')
  imagesB = glob(data_dir + '/testB/*.*')
  
  imagesA = np.random.choice(a = imagesA, size = batch_size)
  imagesB = np.random.choice(a = imagesB, size = batch_size)
  
  allA = []
  allB = []
  
  for i in range(len(imagesA)):
    ## Load and resize images
    imgA = resize(imread(imagesA[i], pilmode = 'RGB').astype(np.float32), (128, 128))
    imgB = resize(imread(imagesB[i], pilmode = 'RGB').astype(np.float32), (128, 128))
    
    allA.append(imgA)
    allB.append(imgB)
    
  return np.array(allA) / 127.5 - 1.0, np.array(allB) / 127.5 - 1.0
def write_log(callback, name, loss, batch_no):
  
  """
  Write training summary to TensorBoard
  """
  
  # summary = tf.Summary()
  # summary_value = summary.value.add()
  # summary_value.simple_value = loss
  # summary_value.tag = name

  # with self.writer.as_default():
  #   tf.summary.scalar(name, loss)
  #   self.writer.flush()

  # callback.writer.add_summary(summary, batch_no)
  # callback.writer.flush()
  
  # self.writer = tf.summary.create_file_writer(log_dir)
  
if __name__ == '__main__':
  
  data_dir = "../input/monet2photo/monet2photo"
  batch_size = 1
  epochs = 500
  mode = 'train'
  
  if mode == 'train':
    
    """
    Load dataset
    """
    
    imagesA,imagesB = load_images(data_dir = data_dir)
    
    ## Define the common optimizer
    common_optimizer = Adam(0.002, 0.5)
    
    
    ## Build and compile discriminator networks
    discriminatorA = build_discriminator()
    discriminatorB = build_discriminator()
    
    discriminatorA.compile(loss = 'mse', 
                           optimizer = common_optimizer,
                           metrics = ['accuracy'])
    discriminatorB.compile(loss = 'mse',
                           optimizer = common_optimizer,
                           metrics = ['accuracy'])
    
    
    ## Build generator networks
    generatorA_to_B = build_generator()
    generatorB_to_A = build_generator()
    
    
    """
    Create an adversarial network
    """
    
    inputA = Input(shape = (128, 128, 3))
    inputB = Input(shape = (128, 128, 3))
    
    
    ## --> Generated images using both of the generator networks
    generatedB = generatorA_to_B(inputA)
    generatedA = generatorB_to_A(inputB)
    
    
    ## --> Reconstruct the images back to the original ones
    reconstructedA = generatorB_to_A(generatedB)
    reconstructedB = generatorA_to_B(generatedA)
    
    generatedA_Id = generatorB_to_A(inputA)
    generatedB_Id = generatorA_to_B(inputB)
    
    
    
    ## Make both of the discriminator networks non-trainable
    discriminatorA.trainable=False
    discriminatorB.trainable=False
    
    
    probsA = discriminatorA(generatedA)
    probsB = discriminatorB(generatedB)
    
    
    adversarial_model = Model(inputs = [inputA, inputB],
                              outputs = [probsA, probsB, 
                                         reconstructedA, reconstructedB,
                                         generatedA_Id, generatedB_Id])
    
    adversarial_model.compile(loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights = [1, 1, 10.0, 10.0, 1.0, 1.0],
                              optimizer = common_optimizer)
    
    
    tensorboard = TensorBoard(log_dir = "../output/logs/{}".format(time.time()),
                              write_images = True, write_grads = True,
                              write_graph = True)
    tensorboard.set_model(generatorA_to_B)
    tensorboard.set_model(generatorB_to_A)
    tensorboard.set_model(discriminatorA)
    tensorboard.set_model(discriminatorB)
    
    
    real_labels = np.ones((batch_size, 7, 7, 1))
    fake_labels = np.zeros((batch_size, 7, 7, 1))
    
    for epoch in range(epochs):
      print("Epoch: {}".format(epoch))
      
      D_losses = []
      G_losses = []
      
      num_batches = int(min(imagesA.shape[0], imagesB.shape[0]) / batch_size)
      print("Number of batches: {}".format(num_batches))
      
      
      for index in range(num_batches):
        print("Batch: {}".format(index))
        
        ## Sample images
        batchA = imagesA[index * batch_size: (index + 1) * batch_size]
        batchB = imagesB[index * batch_size: (index + 1) * batch_size]
        
        ## Translate images to opposite domain
        generatedB = generatorA_to_B.predict(batchA)
        generatedA = generatorB_to_A.predict(batchB)
        
        ## Train the discriminator A on real and fake images
        D_A_Loss1 = discriminatorA.train_on_batch(batchA, np.ones((batch_size, 7, 7, 1)))
        D_A_Loss2 = discriminatorA.train_on_batch(generatedA, np.zeros((batch_size, 7, 7, 1)))
        
        ## Train the discriminator B on real and fake images
        D_B_Loss1 = discriminatorB.train_on_batch(batchB, np.ones((batch_size, 7, 7, 1)))
        D_B_Loss2 = discriminatorB.train_on_batch(generatedB, np.zeros((batch_size, 7, 7, 1)))
        
        ## Calculate the total discriminator loss
        D_loss = 0.5 * np.add(0.5 * np.add(D_A_Loss1, D_A_Loss2), 
                              0.5 * np.add(D_B_Loss1, D_B_Loss2))
        
        print("D_Loss: {}".format(D_loss))
        
        
        """
        Train the generator networks
        """
        
        G_loss = adversarial_model.train_on_batch([batchA, batchB],
                                                  [real_labels, real_labels,
                                                   batchA, batchB,
                                                   batchA, batchB])
        
        print("G_Loss: {}".format(G_loss))
        
        D_losses.append(D_loss)
        G_losses.append(G_loss)
        
        
      """
      Save losses to TensorBoard after every epoch
      """
      
      write_log(tensorboard, 'discriminator_loss', np.mean(D_losses), epoch)
      write_log(tensorboard, 'generator_loss', np.mean(G_losses), epoch)
      
      
      ## Sample and save images after every 10 epochs
      if epoch % 1 == 0:
        
        ## Get a batch of test data
        batchA, batchB = load_test_batch(data_dir = data_dir, batch_size = 2)
        """!zip -r /content/file.zip /content/sample_data/gen
        files.download("/content/file.zip")"""
        
        ## Generate images
        generatedB = generatorA_to_B.predict(batchA)
        generatedA = generatorB_to_A.predict(batchB)
        
        ## Get reconstructed images
        recons_A = generatorB_to_A.predict(generatedB)
        recons_B = generatorA_to_B.predict(generatedA)
        
        ## Save original, generated and reconstructed images
        for i in range(len(generatedA)):
          save_images(originalA = batchA[i], generatedB = generatedB[i], reconstructedA = recons_A[i],
                      originalB = batchB[i], generatedA = generatedA[i], reconstructedB = recons_B[i],
                      path ="../kaggle/working/gen/gen_{}_{}".format(epoch, i))
          
    
    ## Save models
    generatorA_to_B.save_weights("generatorA_to_B.h5")
    generatorB_to_A.save_weights("generatorB_to_A.h5")
    discriminatorA.save_weights("discriminatorA.h5")
    discriminatorB.save_weights("discriminatorB.h5")
    
    
  elif mode == 'predict':
    
    ## Build generator networks
    generatorA_to_B = build_generator()
    generatorB_to_A = build_generator()
    
    generatorA_to_B.load_weights("generatorA_to_B.h5")
    generatorB_to_A.load_weights("generatorB_to_A.h5")
    
    
    ## Get a batch of test data
    batchA, batchB = load_test_batch(data_dir = data_dir, batch_size = 2)
    
    
    ## Save images
    generatedB = generatorA_to_B.predict(batchA)
    generatedA = generatorB_to_A.predict(batchB)
    
    reconsA = generatorB_to_A.predict(generatedB)
    reconsB = generatorA_to_B.predict(generatedA)
    
    for i in range(len(generatedA)):
      save_images(originalA = batchA[i], generatedB = generatedB[i], reconstructedA = recons_A[i],
                  originalB = batchB[i], generatedA = generatedA[i], reconstructedB = recons_B[i],
                  path = "../kaggle/working/test/test_{}".format(i))