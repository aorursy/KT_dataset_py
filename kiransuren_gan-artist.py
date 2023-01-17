import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import PIL
import tensorflow as tf
from tensorflow.keras import layers
import time

from IPython import display

tf.__version__
datasetDirectory  = "/kaggle/input/best-artworks-of-all-time/"
resized = datasetDirectory + "resized/resized/"
artist = "Andy_Warhol"                                                                #Change for a different directory
imageDirectory = glob.glob(datasetDirectory+"resized/resized/{}*.jpg".format(artist))
num_paintings = len(imageDirectory)
print("Number of Paintings by {}: {}".format(artist, num_paintings))
#Although these images have been resized to similar sizes, we must further resize them so they are able to go through the GAN
#Another approach is using Spatial Pyramid Pooling
#Find average height and width of all images in dataset
tW = 0
tH = 0
for i in imageDirectory:
    image = PIL.Image.open(i)
    width,height = image.size
    tW += width
    tH += height
avg_width = round(tW/(num_paintings))
avg_height = round(tH/(num_paintings))
print("Average Width of Images", avg_width)
print("Average Heigth of Images", avg_height)
def decodeImage(i, scale = False):
    """
    Decodes an image, by loading the jpg from the directory, decoding the jpg into a uint8 tensor (RGB channels), then converting to a float32 tensor
    """
    img = tf.io.read_file(i)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if(scale):
        img = tf.image.resize(img, [avg_width,avg_height])  
    return img

def showImage(i):
    """
    Shows image, converts float32 tensor to uint8 tensor, then to a numpy array to display
    """
    i = tf.image.convert_image_dtype(i, tf.uint8)
    i = PIL.Image.fromarray(i.numpy())
    display.display(i)
#SANITY CHECK
for i in range(3):
    showImage(decodeImage(imageDirectory[i]))
#Convert list of directories to a Dataset of tensor32
dirSet = tf.data.Dataset.from_tensor_slices(imageDirectory)
imageSet = dirSet.map(decodeImage)

#More image prepreocessing

BATCH_SIZE = 10
BUFFER_SIZE = 40          #We won't use this if we are taking the Spatial Pyramid Approach
imageSet = imageSet.shuffle(BUFFER_SIZE)
#SANITY CHECK
for i in imageSet.take(2):
    showImage(i)
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Conv2DTranspose

#Since the artworks have a variable size, the random input will be scaled by some factor to give the final dimensions of the generated artwork


class GeneratorNetwork(tf.keras.Model):
    
    def __init__(self):
        super(GeneratorNetwork, self).__init__()
        
        self.dA = Dense(50*50*1024)
        self.bA = BatchNormalization()
        self.actA = LeakyReLU()
        
        self.shaped = Reshape((50,50,1024))
        
        self.conv1 = Conv2DTranspose(512, (5,5), strides = (1,1),  padding='same', use_bias=False)  
        self.b1 = BatchNormalization()
        self.act1 = LeakyReLU()
        
        self.conv2 = Conv2DTranspose(256, (5,5), strides = (2,2),  padding='same', use_bias=False)  
        self.b2 = BatchNormalization()
        self.act2 = LeakyReLU()
        
        self.conv3 = Conv2DTranspose(128, (5,5), strides = (2,2),  padding='same', use_bias=False)  
        self.b3 = BatchNormalization()
        self.act3 = LeakyReLU()
        
        self.conv4 = Conv2DTranspose(64, (5,5), strides = (2,2),  padding='same', use_bias=False)  
        self.b4 = BatchNormalization()
        self.act4 = LeakyReLU()
        
        self.conv5 = Conv2DTranspose(3, (5,5), strides = (1,1),  padding='same', use_bias=False)  
        self.b5 = BatchNormalization()
        self.act5 = LeakyReLU()
    
    
    def call(self, x):
        
        x = self.dA(x) 
        x = self.bA(x) 
        x = self.actA(x) 
        
        x = self.shaped(x) 
        
        x = self.conv1(x) 
        x = self.b1(x) 
        x = self.act1(x) 
        
        x = self.conv2(x) 
        x = self.b2(x) 
        x = self.act2(x) 
        
        x = self.conv3(x) 
        x = self.b3(x) 
        x = self.act3(x) 
        
        x = self.conv4(x) 
        x = self.b4(x) 
        x = self.act4(x) 
        
        x = self.conv5(x) 
        x = self.b5(x) 
        x = self.act5(x) 
    
        return x
generator = GeneratorNetwork()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 1], cmap='gray')
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, LeakyReLU, Concatenate, SeparableConv2D

ACT = 'Leakyelu'

stride =1

class DiscriminatorNetwork(tf.keras.Model):
    
    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()
        
        #Primary Convolutional Layers
        self.conv1 = Conv2D(64,(3,3), stride, padding='same', input_shape= (None,None,3))
        self.conv2 = Conv2D(64,(3,3), stride, padding='same' )
        self.conv3 = Conv2D(64,(3,3), stride, padding='same')
        self.act1 =  LeakyReLU()
        #self.pool1 = MaxPool2D(pool_size =(2,2))
        
        self.conv4 = SeparableConv2D(128,(3,3), stride, padding='same')
        self.conv5 = SeparableConv2D(128,(3,3), stride, padding='same')
        self.conv6 = SeparableConv2D(128,(3,3), stride, padding='same')
        self.act2 =  LeakyReLU()
        #self.pool2 = MaxPool2D(pool_size = (2,2))
        
        self.conv7 = SeparableConv2D(256,(3,3), 2, padding='same')
        self.conv8 = SeparableConv2D(256,(3,3), 2, padding='same')
        self.conv9 = SeparableConv2D(256,(3,3), 2, padding='same')
        self.act3 =  LeakyReLU()
        self.pool3 = MaxPool2D(pool_size = (2,2))
        
        self.conv10 = SeparableConv2D(512,(3,3), stride, padding='same')
        self.conv11 = SeparableConv2D(512,(3,3), stride, padding='same')
        self.conv12 = SeparableConv2D(512,(3,3), stride, padding='same')
        self.act4 =  LeakyReLU()
        #self.pool4 = MaxPool2D(pool_size = (2,2))
        
        #Spatial Pyramid Layers
        self.pyramid1 = MaxPool2D(pool_size = (1,1))
        self.pyramid2 = Flatten()
        self.pyramid3 = MaxPool2D(pool_size = (2,2))
        self.pyramid4 = Flatten()
        self.pyramid5 = MaxPool2D(pool_size = (4,4))
        self.pyramid6 = Flatten()
        
        self.pyramidConcat = Concatenate()
        
        #Fully Connected Layers
        self.dense1 = Dense(1024, activation='relu')
        self.drop1 = Dropout(0.7)
        self.dense2 = Dense(512, activation='relu')
        self.drop2 = Dropout(0.5)
        self.denseOUT = Dense(1)
        
    
    def call(self, x):
        
        
        #Primary Convolutional Layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act1(x)
        #x = self.pool1(x)
        
        
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.act2(x)
        #x = self.pool2(x)
        
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.act3(x)
        x = self.pool3(x)
        
        
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.act4(x)
        #x = self.pool4(x)
        
        #Spatial Pyramid Layers
        a = self.pyramid1(x)
        a = self.pyramid2(a)
        
        b = self.pyramid3(x)
        b = self.pyramid4(b)
        
        c = self.pyramid5(x)
        c = self.pyramid6(c)

        out = self.pyramidConcat([a,b,c])
        
        #Fully Connected Layers
        
        out = self.dense1(out)
        out = self.drop1(out)
        out = self.dense2(out)
        out = self.drop2(out)
        out = self.denseOUT(out)
        
        return out
discriminator = DiscriminatorNetwork()
decision = discriminator(generated_image)
print (decision)
generator = GeneratorNetwork()
discriminator = DiscriminatorNetwork()
#Loss functions using cross entropy
# TODO: Loss functions based off Wasserstein Loss

LEARNING_RATE  = 0.001

cr = tf.keras.losses.BinaryCrossentropy(from_logits=True)

"""
def discriminator_loss(real, fake):
    critic_loss =  -1 * (real - fake)     #Add -1 to make a minimization problem
    return critic_loss

def generator_loss(fake):
    return -1 *fake

"""
def disc_loss(real, fake):
    real_l  = cr(tf.ones_like(real), real)   #Compares tensor of ones to real output  (closer real is to 1, lower the loss)
    fake_l  = cr(tf.zeroes_like(fake), fake)   #Compares tensor of zeros to real output  (closer fake is to zero, lower the loss)
    return real_l + fake_l

def gen_loss(fake):
    return cr(tf.ones_like(fake), fake) #Closer the fake is to 1, lower the loss (means that the discriminator thinks the fakes are real)


disc_opt = tf.keras.optimizers.Adam(LEARNING_RATE)
gen_opt = tf.keras.optimizers.Adam(LEARNING_RATE)

#Create checkpoints for saving model
checkpoint_dir = './training_chckpts'
checkpoint_pref = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(gen_opt=gen_opt,
                                 disc_opt=disc_opt,
                                 generator=generator,
                                 discriminator=discriminator)
EPOCHS = 50
noise_dim = 100
examples = 3

#Seed to monitor generate image during training
seed = tf.random.normal([examples, noise_dim])
#Create training functions of generator and discriminator
@tf.function
def train_stepGenerator(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as tape:
      generated_images = generator(noise, training=True)
      critique = discriminator(generated_images, training=True)
      loss = gen_loss(critique)

    grads = tape.gradient(loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(grads, generator.trainable_variables))
    
@tf.function
def train_stepDiscriminator(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as tape:
      generated_images = generator(noise, training=True)
      critique = discriminator(generated_images, training=True)
      loss = disc_loss(image, critique)

    grads = tape.gradient(loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(grads, discriminator.trainable_variables))
epoch = EPOCHS
for i in range(epoch):
    start = time.time()
    
    #Even epoch
    if(epoch % 2 == 0):
        for image in imageSet:
            train_stepGenerator(image)
    else:
        for images in imageSet:
            train_stepDiscriminator(image)
            
    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    display.clear_output(wait=True)
    for i in range(len(seed)):
        showImage(generator(seed[i]))
    