!pip install git+https://github.com/crowdai/coco.git#subdirectory=PythonAPI
# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Garbage collector

import gc



# Folder manipulation

import os

from glob import glob



# Linear algebra

import numpy as np



# Visualisation of picture and graph

from matplotlib import pylab as plt

import cv2



# Get version python/keras/tensorflow/sklearn

from platform import python_version

import sklearn

import keras

import tensorflow as tf



# Keras importation

import keras.layers as layers

from keras.models import Model

from keras.optimizers import Adam



# Plotting segmentation mask

from pycocotools.coco import COCO

from pycocotools import mask as cocomask

import skimage.io as io



# Others import

import random

import time
print(os.listdir("../input"))

print("Keras version : " + keras.__version__)

print("Tensorflow version : " + tf.__version__)

print("Python version : " + python_version())

print("Sklearn version : " + sklearn.__version__)
IMG_ROWS = 256

IMG_COLS = 256

CHANNELS = 3

MAX_IMAGES = 3000 # Max number of picture used for training

# (more create a ressource crash on Kaggle...)



MAIN_DIR = "../input/"

IMAGES_DIR = f"{MAIN_DIR}train/train/images/"

ANNOTATIONS_PATH = f"{MAIN_DIR}train/train/annotation-small.json"



plt.rcParams['figure.figsize'] = (8.0, 10.0) # Set images sizes for plotting result during learning
def load_random_data():

    coco = COCO(ANNOTATIONS_PATH)

    

    image_ids = coco.getImgIds(catIds=coco.getCatIds())

    

    # Select a random pictures id

    random_image_id = random.choice(image_ids)

    img = coco.loadImgs(random_image_id)[0]



    annotation_ids = coco.getAnnIds(imgIds=img['id'])

    annotations = coco.loadAnns(annotation_ids)



    image_path = os.path.join(IMAGES_DIR, img["file_name"])

    I = io.imread(image_path) # Image en png



    mask = np.zeros((300, 300))

    for _idx, annotation in enumerate(annotations):

        rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])

        m = cocomask.decode(rle)

        m = m.reshape((img['height'], img['width']))

        mask = mask + m

    

    return I, mask
img, mask = load_random_data()
def plot_pictures(img, mask):

    fig, axs = plt.subplots(1,2, figsize=(7.5, 7.5))

        

    axs[0].imshow(img)

    axs[0].set_title("Picture")

    axs[0].axis('off')

    

    axs[1].imshow(mask)

    axs[1].set_title("Mask")

    axs[1].axis('off')
plot_pictures(img, mask)
def load_data(batch_size):

    coco = COCO(ANNOTATIONS_PATH)

    

    path1 = sorted(glob(IMAGES_DIR + "*"))

    path2 = coco.getImgIds(catIds=coco.getCatIds())

    

    i = np.random.randint(0,27)

    batch1 = path1[i*batch_size:(i+1)*batch_size]

    batch2 = path2[i*batch_size:(i+1)*batch_size]

    

    img_A, img_B = [], []

    

    for filename1,filename2 in zip(batch1,batch2):

        

        json = coco.loadImgs(filename2)[0]

        image_path = os.path.join(IMAGES_DIR, json["file_name"])

        img = cv2.imread(image_path)

        

        annotation_ids = coco.getAnnIds(imgIds=json['id'])

        annotations = coco.loadAnns(annotation_ids)

        

        mask = np.zeros((300, 300, 1))

        

        for _idx, annotation in enumerate(annotations):

            rle = cocomask.frPyObjects(annotation['segmentation'], json['height'], json['width'])

            m = cocomask.decode(rle)

            mask = mask + m

        

        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)

        mask = cv2.resize(mask,(256,256),interpolation=cv2.INTER_AREA)

        

        mask = mask * 255

        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

        

        img_A.append(img)

        img_B.append(mask)

      

    img_A = np.array(img_A) / 127.5 - 1

    img_B = np.array(img_B) / 127.5 - 1

    

    return img_A, img_B 
def load_batch(batch_size):

    coco = COCO(ANNOTATIONS_PATH)

    

    path1 = sorted(glob(IMAGES_DIR + "*"))

    path2 = coco.getImgIds(catIds=coco.getCatIds())

    

    n_batches=int(len(path1)/batch_size)

    max_batch = 0

  

    for i in range(n_batches):

        batch1 = path1[i*batch_size:(i+1)*batch_size]

        batch2 = path2[i*batch_size:(i+1)*batch_size]

        img_A, img_B=[],[]

        

        for filename1,filename2 in zip(batch1,batch2):

            json = coco.loadImgs(filename2)[0]

            image_path = os.path.join(IMAGES_DIR, json["file_name"])

            img1 = cv2.imread(image_path)

            

            annotation_ids = coco.getAnnIds(imgIds=json['id'])

            annotations = coco.loadAnns(annotation_ids)



            mask = np.zeros((300, 300, 1))

            for _idx, annotation in enumerate(annotations):

                rle = cocomask.frPyObjects(annotation['segmentation'], json['height'], json['width'])

                m = cocomask.decode(rle)

                mask = mask + m



            img1 = img1

            img2 = mask



            img1=cv2.resize(img1,(256,256),interpolation=cv2.INTER_AREA)

            img2=cv2.resize(img2,(256,256),interpolation=cv2.INTER_AREA)



            img2 = np.reshape(img2, (img2.shape[0], img2.shape[1], 1))

            

            img_A.append(img1)

            img_B.append(img2*255)



        img_A=np.array(img_A) / 127.5-1

        img_B=np.array(img_B) / 127.5-1

        

        max_batch = max_batch + 1

        

        if(max_batch > MAX_IMAGES):

            raise StopIteration

        else:

            yield img_B, img_A
class Pix2pix():

    def __init__(self):

        self.img_rows = IMG_ROWS

        self.img_cols = IMG_COLS

        self.channels = CHANNELS

        self.img_shape = (self.img_rows,self.img_cols,self.channels)

        self.mask_shape = (self.img_rows,self.img_cols,1)

    

        patch = int(self.img_rows / (2**4)) # 16

        self.disc_patch = (patch, patch, 1)



        self.gf = 64

        self.df = 64

    

        optimizer = Adam(0.0002,0.5)

    

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',

                              optimizer=optimizer)

    

        self.generator = self.build_generator()

    

        mask_input = layers.Input(shape=self.mask_shape) #img_B = layers.Input(shape=self.img_shape)

        

        img = self.generator(mask_input)

    

        self.discriminator.trainable = False

    

        valid = self.discriminator([img, mask_input])

    

        self.combined = Model(mask_input, valid)

        self.combined.compile(loss='binary_crossentropy',

                              optimizer=optimizer)

    

    def build_generator(self):

        def conv2d(layer_input,filters,f_size=(4,4),bn=True):

            d = layers.Conv2D(filters,kernel_size=f_size,strides=(2,2),padding='same')(layer_input)

            d = layers.LeakyReLU(0.2)(d)

            if bn:

                d = layers.BatchNormalization()(d)

            return d

    

        def deconv2d(layer_input,skip_input,filters,f_size=(4,4),dropout_rate=0):

            u = layers.UpSampling2D((2,2))(layer_input)

            u = layers.Conv2D(filters,kernel_size=f_size,strides=(1,1),padding='same',activation='relu')(u)

            if dropout_rate:

                u = layers.Dropout(dropout_rate)(u)

            u = layers.BatchNormalization()(u)

            u = layers.Concatenate()([u,skip_input])

            return u

    

        d0 = layers.Input(shape=self.mask_shape)

    

        d1 = conv2d(d0,self.gf,bn=False) 

        d2 = conv2d(d1,self.gf*2)         

        d3 = conv2d(d2,self.gf*4)         

        d4 = conv2d(d3,self.gf*8)         

        d5 = conv2d(d4,self.gf*8)         

        d6 = conv2d(d5,self.gf*8)        

    

        d7 = conv2d(d6,self.gf*8)         

    

        u1 = deconv2d(d7,d6,self.gf*8,dropout_rate=0.5)   

        u2 = deconv2d(u1,d5,self.gf*8,dropout_rate=0.5)   

        u3 = deconv2d(u2,d4,self.gf*8,dropout_rate=0.5)   

        u4 = deconv2d(u3,d3,self.gf*4)   

        u5 = deconv2d(u4,d2,self.gf*2)   

        u6 = deconv2d(u5,d1,self.gf)     

        u7 = layers.UpSampling2D((2,2))(u6)

    

        output_img = layers.Conv2D(3,kernel_size=(4,4),strides=(1,1),padding='same',activation='tanh')(u7)

    

        return Model(d0,output_img)

  

    def build_discriminator(self):

        def d_layer(layer_input,filters,f_size=(4,4),bn=True):

            d = layers.Conv2D(filters,kernel_size=f_size,strides=(2,2),padding='same')(layer_input)

            d = layers.LeakyReLU(0.2)(d)

            if bn:

                d=layers.BatchNormalization()(d)

            return d

    

        img_input = layers.Input(shape=self.img_shape)

        mask_input = layers.Input(shape=self.mask_shape)

    

        combined_imgs = layers.Concatenate(axis=-1)([img_input, mask_input])

    

        d1 = d_layer(combined_imgs,self.df,bn=False)

        d2 = d_layer(d1,self.df*2)

        d3 = d_layer(d2,self.df*4)

        d4 = d_layer(d3,self.df*8)

    

        validity = layers.Conv2D(1,kernel_size=(4,4),strides=(1,1),padding='same',activation='sigmoid')(d4)

    

        return Model([img_input, mask_input], validity)

  

    def train(self,epochs,batch_size=1):

        valid=np.ones((batch_size,)+self.disc_patch)

        fake=np.zeros((batch_size,)+self.disc_patch)

    

        for epoch in range(epochs):

            start=time.time()

            for batch_i,(img_A,img_B) in enumerate(load_batch(1)):

                gen_imgs=self.generator.predict(img_A)

        

                d_loss_real = self.discriminator.train_on_batch([img_B, img_A], valid)

                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, img_A], fake)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        

                g_loss = self.combined.train_on_batch(img_A,valid)



                if batch_i % 50 == 0:

                    print ("[Epoch %d] [Batch %d] [D loss: %f] [G loss: %f]" % (epoch,

                                                                                batch_i,

                                                                                d_loss,

                                                                                g_loss))

            

            self.sample_images(epoch)

            print('Time for epoch {} is {} sec'.format(epoch,time.time()-start))

      

    def sample_images(self, epoch):

        # Set values

        row, col = 3, 3

        nb_img = 3

        

        # Load and generate data

        img_A, img_B =load_data(nb_img)

        fake_A = self.generator.predict(img_B)

        

        # Rescale or reshape data

        img_A = 0.5 * img_A + 0.5

        fake_A = 0.5 * fake_A + 0.5

        img_B = np.reshape(img_B, (3, 256, 256)) # Avoid matplotlib error

        

        imgs = [img_A, fake_A, img_B]



        # Plot results pictures

        titles = ['Condition', 'Generated', 'Original']

        

        fig, axs = plt.subplots(row, col)

        

        for r, img, title in zip(range(0, row), imgs, titles):

            for c in range(0, col):

                axs[r,c].imshow(img[c])

                axs[r,c].set_title(title)

                axs[r,c].axis('off')

        

        plt.show()
gan = Pix2pix()

gan.train(epochs=70, batch_size=1)