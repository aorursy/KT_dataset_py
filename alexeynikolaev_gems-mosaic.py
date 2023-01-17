import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K

import os
import argparse
import glob 

from PIL import Image
import matplotlib.pyplot as plt

import sys

import numpy as np

TRANSPARENT_GEMS = [
    'Garnet Red', 'Diamond', 'Quartz Rutilated', 'Topaz', 'Quartz Smoky', 'Citrine', 'Tanzanite', 'Tsavorite',
    'Hessonite', 'Quartz Lemon', 'Peridot', 'Ametrine', 'Sphene', 'Morganite', 'Zircon', 'Grossular', 'Benitoite',
    'Diaspore', 'Rhodolite', 'Iolite', 'Pyrope', 'Amethyst', 'Almandine', 'Sapphire Blue', 'Spessartite',
    'Chrome Diopside', 'Tourmaline', 'Quartz Beer', 'Chrysoberyl', 'Sapphire Yellow', 'Andradite', 'Kyanite',
    'Andalusite', 'Beryl Golden', 'Danburite', 'Kunzite', 'Quartz Rose', 'Sapphire Pink', 'Aquamarine',
    'Sapphire Purple', 'Alexandrite', 'Spodumene', 'Ruby', 'Emerald', 'Hiddenite', 'Goshenite', 'Bixbite'
]

class DCGAN():
    def __init__(self, img_rows=128, img_cols=128, channels=4, latent_dim=3, loss='binary_crossentropy', name='earth'):
        self.name = name

        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim
        self.loss = loss

        self.optimizer = Adam(0.0005, 0.6)
        #self.optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

        # Build the GAN
        self.build_combined()
        
    def build_combined(self):
        self.discriminator.compile(loss='binary_crossentropy',
                optimizer=self.optimizer,
                metrics=['accuracy'])
        
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer)    
    
    def load_weights(self, generator_file=None, discriminator_file=None):

        if generator_file:
            generator = self.build_generator()
            generator.load_weights(generator_file)
            self.generator = generator
            print('generator weights loaded')
    
        if discriminator_file:
            discriminator = self.build_discriminator()
            discriminator.load_weights(discriminator_file)
            self.discriminator = discriminator
            print('discriminator weights loaded')

        if generator_file or discriminator_file: 
            self.build_combined() 
            print('build compaied ')

    def build_generator(self):

        model = Sequential()
        #model.add(Dense(128, activation="relu", input_dim=self.latent_dim, name="generator_input") )
        #model.add(Dropout(0.1))
        
        #model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim, name="generator_input") )
        model.add(Dense(128 * 32 * 32, activation="relu", input_dim=self.latent_dim, name="generator_input"))
        model.add(Dropout(0.1))
        #model.add(Reshape((16, 16, 128)))
        model.add(Reshape((32, 32, 128)))
        #model.add(UpSampling2D())

        model.add(Conv2D(128, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(UpSampling2D())
        
        model.add(Conv2D(128, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        #model.add(UpSampling2D())

        model.add(Conv2D(self.channels, kernel_size=3, padding="same", activation="sigmoid", name="generator_output"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img, name="generator")

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())

        #model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        discrim = Model(img, validity)

        return discrim

    def train(self, X_train, epochs, batch_size=128, save_interval=100):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            if epoch % 10 == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                if not os.path.exists('/kaggle/working/images'):
                    os.mkdir('/kaggle/working/images')
        
                self.save_imgs( "/kaggle/working/images/{}_{:05d}.png".format(self.name,epoch) )
                # self.combined.save_weights("combined_weights ({}).h5".format(self.name)) # https://github.com/keras-team/keras/issues/10949
                self.generator.save_weights("/kaggle/working/generator ({}).h5".format(self.name))
                self.discriminator.save_weights("/kaggle/working/discriminator ({}).h5".format(self.name))

    def save_imgs(self, name=''):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        # replace the first two latent variables with known values
        #for i in range(r):
        #    for j in range(c):
        #        noise[4*i+j][0] = i/(r-1)-0.5
        #        noise[4*i+j][1] = j/(c-1)-0.5

        gen_imgs = self.generator.predict(noise)

        fig, axs = plt.subplots(r, c, figsize=(6.72,6.72))
        plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95, wspace=0.2, hspace=0.2)

        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1

        if name:
            fig.savefig(name, facecolor='black' )
        else: 
            fig.savefig('{}.png'.format(self.name), facecolor='black' )

        plt.close()
    

def export_model(saver, model, model_name, input_node_names, output_node_name):
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.tools import optimize_for_inference_lib
    
    if not os.path.exists('/kaggle/working/out'):
        os.mkdir('/kaggle/working/out')

    tf.train.write_graph(K.get_session().graph_def, '/kaggle/working/out', model_name + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('/kaggle/working/out/' + model_name + '_graph.pbtxt', None, False,
                              '/kaggle/working/out/' + model_name + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              '/kaggle/working/out/frozen_' + model_name + '.bytes', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('/kaggle/working/out/frozen_' + model_name + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('/kaggle/working/out/opt_' + model_name + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")
import numpy as np
import pandas as pd

import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
train_images = []
for dirname, _, filenames in os.walk('/kaggle/input/gemstones-images/train'):
    if dirname.split('/')[-1] in TRANSPARENT_GEMS:
        for filename in filenames:
            train_images.append(os.path.join(dirname, filename))
        
#test_images = []
for dirname, _, filenames in os.walk('/kaggle/input/gemstones-images/test'):
    if dirname.split('/')[-1] in TRANSPARENT_GEMS:
        for filename in filenames:
            train_images.append(os.path.join(dirname, filename))
SCALE = 128

all_img = []
for image in train_images:
        _,extension = os.path.splitext(image)
        if(extension==".jpg" or extension==".jpeg" or extension==".png"):
            img=load_img(image)
            img=img.resize((SCALE,SCALE), Image.LANCZOS)
            x=img_to_array(img)
            all_img.append(x)

'''
all_img_test = []
for image in test_images:
        _,extension = os.path.splitext(image)
        if(extension==".jpg" or extension==".jpeg" or extension==".png"):
            img=load_img(image)
            img=img.resize((SCALE,SCALE), Image.LANCZOS)
            x=img_to_array(img)
            all_img_test.append(x)   
'''
all_img=np.asarray(all_img,dtype="float")
#all_img_test=np.asarray(all_img_test,dtype="float")

train=all_img/255
#test=all_img_test/255
fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        axs[i,j].imshow( train[ np.random.randint(train.shape[0]) ] )
        axs[i,j].axis('off')
plt.show()
dcgan = DCGAN(img_rows = train[0].shape[0],
                    img_cols = train[0].shape[1],
                    channels = train[0].shape[2], 
                    latent_dim = 256,
                    name='gems_128_256_new')
dcgan.train(train, epochs=30000, batch_size=32, save_interval=500)
dcgan = DCGAN(img_rows = train[0].shape[0],
                    img_cols = train[0].shape[1],
                    channels = train[0].shape[2], 
                    latent_dim = 256)
dcgan.load_weights(
    generator_file='/kaggle/input/gem-gans/generator (gems_256_128_30000).h5',
    discriminator_file='/kaggle/input/gem-gans/discriminator (gems_256_128_30000).h5'
)
noise = np.random.normal(0, 1, (16, dcgan.latent_dim))
gen_imgs = dcgan.generator.predict(noise)
fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        axs[i,j].imshow(gen_imgs[i*4+j])
        axs[i,j].axis('off')
plt.show()
plt.imshow(gen_imgs[0])
import matplotlib
import cv2

def remove_background(image_array):
    matplotlib.image.imsave("/kaggle/working/image1.png", image_array)
    
    #== Parameters           
    BLUR = 21
    CANNY_THRESH_1 = 100
    CANNY_THRESH_2 = 220
    MASK_DILATE_ITER = 5
    MASK_ERODE_ITER = 7
    MASK_COLOR = (0.0,0.0,0.0) # In BGR format


    #-- Read image
    img = cv2.imread("/kaggle/working/image1.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #-- Edge detection 
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    #-- Find contours in edges, sort by area 
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    #-- Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    #-- Blend masked img into MASK_COLOR background
    mask_stack  = mask_stack.astype('float32') / 255.0         
    img         = img.astype('float32') / 255.0    
    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)  
    masked = (masked * 255).astype('uint8')
    #matplotlib.image.imsave("/kaggle/working/image2.png", cv2.cvtColor(masked,cv2.COLOR_BGR2RGB))
    
    # split image into channels
    c_red, c_green, c_blue = cv2.split(img)

    # merge with mask got on one of a previous steps
    img_a = cv2.merge((c_blue, c_green, c_red, mask.astype('float32') / 255.0))
    #matplotlib.image.imsave("/kaggle/working/image2.png", img_a)

    return img_a, masked
from pylab import rcParams
rcParams['figure.figsize'] = 9, 9
from PIL import Image

# Open Paddington
img = Image.open("/kaggle/input/art-and-nature-pics/lavender.jpg")

# Resize smoothly down to 16x16 pixels
imgSmall = img.resize((round(img.width/200),round(img.height/200)),resample=Image.BILINEAR)

# Scale back up using NEAREST to original size
result = imgSmall.resize(img.size,Image.NEAREST)

# Save
result.save('/kaggle/working/pixelated.png')
plt.imshow(imgSmall)
imgArr = np.array(imgSmall)
imgArr.shape[0] * imgArr.shape[1]
color = imgArr[6,6]
!pip install colorthief
from colorthief import ColorThief
color_thief = ColorThief('/kaggle/working/image2.png')
dominant_color = color_thief.get_color(quality=1)
dominant_color
from IPython import display

TARGET = 1600
BATCH_SIZE = 9
counter = 495
gem_colors = []

while counter<TARGET:
    print(counter)
    noise = np.random.normal(0, 1, (BATCH_SIZE, dcgan.latent_dim))
    gen_imgs = dcgan.generator.predict(noise)
    clean_imgs = []
    black_imgs = []
    for j in range(BATCH_SIZE):
        img = gen_imgs[j]
        img_clean, img_black = remove_background(img)
        clean_imgs.append(img_clean)
        black_imgs.append(img_black)
            
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            axs[i,j].imshow(black_imgs[i*3+j])
            axs[i,j].axis('off')
    plt.show()
    
    selected = input()
    if len(selected)>0:
        for s in map(int, selected.split(',')):
            matplotlib.image.imsave("/kaggle/working/gem_{}.png".format(counter), clean_imgs[s-1])
            counter+=1
    
    display.clear_output(wait=True)
!pip install colorthief
from colorthief import ColorThief

gems = []
gems_colors = []
for dirname, _, filenames in os.walk('/kaggle/input/gems-curated'):
    for filename in filenames:
        gems.append(filename)
        color_thief = ColorThief(filename)
        dominant_color = color_thief.get_color(quality=1)
        gems_colors.append(dominant_color)
matches = np.zeros((imgArr.shape[0], imgArr.shape[1]), dtype=int)
taken = []

for i in range(imgArr.shape[0]):
    for j in range(imgArr.shape[1]):
        color = imgArr[i,j]
        #if sum(color) < 150:
            #matches[i, j] = -1
            #continue
        min_dist = np.inf
        for jj, c in enumerate(gems_colors):
            if jj in taken:
                continue
            else:
                dist = (((color[0] - c[0])**2 + (color[1] - c[1])**2 + (color[2] - c[2])**2)*1.0)**0.5
                if dist < min_dist:
                    min_index = jj
                    min_dist = dist
        matches[i,j] = min_index
        taken.append(min_index)
matches
final = Image.new('RGB', (imgArr.shape[1] * 256, imgArr.shape[0] * 256), (32, 32, 32))
for i in range(imgArr.shape[0]):
    for j in range(imgArr.shape[1]):
        match = matches[i,j]
        if match == -1:
            continue        
        match_img = Image.open(gems[match])
        final.paste(match_img, (j*256, i*256), match_img)

final.save("/kaggle/working/final.png")
def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

print(rgb_to_hsv(255, 255, 255))
print(rgb_to_hsv(0, 215, 0))
gems_hues = list(map(lambda c: rgb_to_hsv(*c)[0], gems_colors))
gems_sorted = [x for _,x in sorted(zip(gems_hues,gems))]
len(gems_sorted)**0.5
final = Image.new('RGB', (192 * 22, 192 * 22), (255, 255, 255))
for i in range(22):
    for j in range(22):     
        match_img = Image.open(gems_sorted[22*i+j])
        final.paste(match_img, (32+j*192, 32+i*192), match_img)
        
final.save("/kaggle/working/final.png")
im = Image.open("/kaggle/working/final.png")
imArr = np.array(im)
for i in range(22-5):
    for j in range(22-5):
        img_part = imArr[i*192:(i+5)*192, j*192:(j+5)*192]
        matplotlib.image.imsave("/kaggle/working/final_{}_{}.png".format(i,j), img_part)