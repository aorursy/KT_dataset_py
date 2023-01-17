import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from PIL import Image
import random

from IPython import display
def loadImage( infilename ) :
    img = Image.open(infilename)
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

#load face data (13233, 90x150 size images)

dataDir = r"../input/croppedfaces/cropped faces"
train_images = np.empty([13000, 150, 90])
test_images = np.empty([233, 150, 90])


n = 0
for file in os.listdir(dataDir):
    img = loadImage(dataDir + "/" + file)
    if n < 13000:
        train_images[n] = img
    else:
        test_images[n - 13000] = img
    n += 1

train_images = train_images.reshape(train_images.shape[0], 150, 90, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 150, 90, 1).astype('float32')

# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

fig = plt.figure(figsize=(4,4))
for i in range(train_images[0:16].shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(train_images[0:16][i, :, :, 0], cmap='gray')
      plt.axis('off')
plt.show()
TRAIN_BUF = 13000
BATCH_SIZE = 20

TEST_BUF = 233
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)
class CVAE(tf.keras.Model):
  def __init__(self, latent_dim, inference_net=None, generative_net=None):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    if inference_net:
        self.inference_net = inference_net
    else:
        self.inference_net = tf.keras.Sequential(
          [
              tf.keras.layers.InputLayer(input_shape=(150, 90, 1)),
              tf.keras.layers.Conv2D(
                  filters=32, kernel_size=9, strides=(3, 3), activation='sigmoid'),
              tf.keras.layers.Conv2D(
                  filters=64, kernel_size=9, strides=(3, 3), activation='sigmoid'),
              tf.keras.layers.Flatten(),
              # No activation
              tf.keras.layers.Dense(latent_dim + latent_dim),
          ]
        )
    if generative_net:
        self.generative_net = generative_net
    else:
        self.generative_net = tf.keras.Sequential(
            [
              tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
              tf.keras.layers.Dense(units=5*5*32, activation=tf.nn.relu),
              tf.keras.layers.Reshape(target_shape=(5, 5, 32)),
              tf.keras.layers.Conv2DTranspose(
                  filters=64,
                  kernel_size=9,
                  strides=(2, 2),
                  padding="SAME",
                  activation='sigmoid'),
              tf.keras.layers.Conv2DTranspose(
                  filters=64,
                  kernel_size=9,
                  strides=(3, 3),
                  padding="SAME",
                  activation='sigmoid'),
              tf.keras.layers.Conv2DTranspose(
                  filters=32,
                  kernel_size=9,
                  strides=(5, 3),
                  padding="SAME",
                  activation='sigmoid'),
              # No activation
              tf.keras.layers.Conv2DTranspose(
                  filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits
def removeRectangle(images): #takes as input a batch from train_dataset
    def cutRectangle(img): #takes as input a numpy arrays
        result = np.copy(img)
        maxSize = 50
        #randint includes both endpoints
        x = random.randint(0, 150 - maxSize - 1)
        y = random.randint(0, 90 - maxSize - 1)
        a = random.randint(20, maxSize)
        b = random.randint(20, maxSize)

        for i in range(a):
            for j in range(b):
                result[x + i, y + j] = 1

        return result
    
    newImgs = np.copy(images.numpy())
    newImgs = newImgs.reshape(newImgs.shape[0], 150, 90).astype('float32')

    for i in range(newImgs.shape[0]):
        newImgs[i] = cutRectangle(newImgs[i])
        
    newImgs = newImgs.reshape(newImgs.shape[0], 150, 90, 1).astype('float32')
    return tf.convert_to_tensor(newImgs)
optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

@tf.function
def compute_loss(model, x, cutImgs):
  #x is a batch
  mean, logvar = model.encode(cutImgs)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  print(type(x_logit))

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer, cutImgs):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x, cutImgs)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
epochs = 20
latent_dim = 25
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)
def generate_and_save_images(model, epoch, test_input):
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
#test removeRectangle

for elm in train_dataset:
    print(type(elm))
    img = removeRectangle(elm)
    mean, logvar = model.encode(img)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    break
    #test cutRectangle
    
    img = train_images[15].reshape(150, 90).astype('float32')
    
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.show()
    
    img2 = cutRectangle(img)
    
    plt.axis("off")
    plt.imshow(img2, cmap='gray')
    plt.show()
model.inference_net.summary()
model.generative_net.summary()
generate_and_save_images(model, 0, random_vector_for_generation)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    cutImgs = removeRectangle(train_x)
    compute_apply_gradients(model, train_x, optimizer, cutImgs)
  end_time = time.time()

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      cutImgs = removeRectangle(test_x)
      loss(compute_loss(model, test_x, cutImgs))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))
    generate_and_save_images(
        model, epoch, random_vector_for_generation)
#save the model

model.inference_net.save_weights("cvae-inference-weights")
model.generative_net.save_weights("cvae-generative-weights")
#load the model

inf_model = model.inference_net
inf_model.load_weights("../input/mostly-trained-inpainting/cvae-inference-weights")

gen_model = model.generative_net
gen_model.load_weights("../input/mostly-trained-inpainting/cvae-generative-weights")

model = CVAE(latent_dim, inf_model, gen_model)
#run network once on random seed vector to generate images

generate_and_save_images(model, 0, random_vector_for_generation)
#show a specific image

plt.axis("off")
plt.imshow(test_images[69].reshape(150, 90).astype('float32'), cmap='gray')
plt.show()
#doesn't work

tempData = np.empty([2, 150, 90])
tempData = tempData.reshape(tempData.shape[0], 150, 90, 1).astype('float32')

tempData[0] = test_images[69]
tempData[1] = test_images[0]

sampleDataset = tf.data.Dataset.from_tensor_slices(tempData)

for elm in sampleData:
    mean, logvar = model.encode(elm)
    z = model.reparameterize(mean, logvar)
    result = model.decode(z)
    break
    
#maybe use tf.convert_to_tensor(image)
#show an element in the first batch of the test_dataset before and after it is run through the network

for elm in test_dataset:
    img = removeRectangle(elm)
    mean, logvar = model.encode(img)
    z = model.reparameterize(mean, logvar)
    result = model.decode(z, apply_sigmoid=True)
    
    plt.axis("off")
    plt.imshow(img.numpy()[15].reshape(150, 90).astype('float32'), cmap='gray')
    plt.show()
    
    plt.axis("off")
    plt.imshow(result.numpy()[15].reshape(150, 90).astype('float32'), cmap='gray')
    plt.show()
    break
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
plt.imshow(display_image(epochs))
plt.axis('off')# Display images
anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info >= (6,2,0,''):
  display.Image(filename=anim_file)
