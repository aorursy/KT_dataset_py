!pip install git+https://www.github.com/keras-team/keras-contrib.git
#import scipy.misc

from keras.datasets import mnist
# from keras_contrib.layers.normalization import InstanceNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os,cv2
from glob import glob

%matplotlib inline
def noisy(img):    
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],1).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(img,gauss)
    # Display the image
    #cv2.imshow('a',img_gauss)
    cv2.waitKey(0)
    return img_gauss
class DataLoader():
    def __init__(self, img_res=(128, 128)):
        self.img_res = img_res
        

    
    def load_data(self,domaineA=False,domaineB=False, batch_size=1):
        path1=sorted(glob('../input/chest-xray-pneumonia/chest_xray/test/NORMAL/*'))
        path2=sorted(glob('../input/covidct/COVID-CT/CT_COVID/*'))
        
        
        i=np.random.randint(0,27)
        
        batch1=path1[i*batch_size:(i+1)*batch_size]
        batch2=path2[i*batch_size:(i+1)*batch_size]
        imgs=[]
        if domaineA :
            for filename1 in batch1:
                img=cv2.imread(filename1,0)
                img=img[...,::-1]
                img=noisy(img)
                img=cv2.resize(img,self.img_res,interpolation=cv2.INTER_AREA)
                imgs.append(img)
        if domaineB :
            for filename2 in batch2:
                img=cv2.imread(filename2,0)
                img=img[...,::-1]
                img=cv2.resize(img,self.img_res,interpolation=cv2.INTER_AREA)
                imgs.append(img)

        imgs = np.reshape(imgs, (len(imgs), 128, 128, 1))
        imgs=np.array(imgs)/127.5-1

    
        return imgs

    def load_batch(self, batch_size=1):
        path_A = sorted(glob('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/*'))
        path_B = sorted(glob('../input/covidct/COVID-CT/CT_COVID/*'))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for file1, file2 in zip(batch_A, batch_B):
                img1 = cv2.imread(file1,0)
                img2 = cv2.imread(file2,0)
                img1 = img1[...,::-1]
                img2 = img2[...,::-1]
                img1 = noisy(img1)
                img1 = cv2.resize(img1,self.img_res,interpolation=cv2.INTER_AREA)
                img2 = cv2.resize(img2,self.img_res,interpolation=cv2.INTER_AREA)
                imgs_A.append(img1)
                imgs_B.append(img2)
                
            imgs_A = np.reshape(imgs_A, (len(imgs_A), 128, 128, 1))
            imgs_B = np.reshape(imgs_B, (len(imgs_B), 128, 128, 1))
            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B



class CycleGAN():
    def __init__(self, img_rows=128, img_cols=128, channels=1):
        self.img_rows = img_rows 
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (img_rows, img_cols, channels)
        
        self.data_loader = DataLoader( img_res=(self.img_rows, self.img_cols))
        

        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)
        
        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64
        
        # controls how strictly the cycle-consistency loss is enforced. Setting this value higher will ensure that you your
        # original and reconstructed image are as close together as possible.
        self.lambda_cycle = 10.0
        # this value influences how dramatic are the changes—especially early in the training process.
        # Setting a lower value leads to unnecessary changes—e.g. completely inverting the colors early on
        self.lambda_id = 0.9 * self.lambda_cycle
        
        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the discriminators
        self.d_a = self.build_discriminator()
        self.d_b = self.build_discriminator()

        self.d_a.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_b.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Build the generators
        self.g_ab = self.build_generator()
        self.g_ba = self.build_generator()
        
        # Input images from both domains
        img_a = Input(shape=self.img_shape)
        img_b = Input(shape=self.img_shape)
        
        # Translate images to the other domain
        fake_b = self.g_ab(img_a)
        fake_a = self.g_ba(img_b)
        
        # Translate images back to original domain
        recon_a = self.g_ba(fake_b)
        recon_b = self.g_ab(fake_a)
        
        # Identity mapping of images
        img_a_id = self.g_ba(img_a)
        img_b_id = self.g_ab(img_b)
        
        # For the combined model we will only train the generators
        self.d_a.trainable = False
        self.d_b.trainable = False
        
        # Discriminators determines validity of translated images
        valid_a = self.d_a(fake_a)
        valid_b = self.d_b(fake_b)
        
        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_a, img_b], output=[valid_a, valid_b, recon_a, recon_b, img_a_id, img_b_id])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                             loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id], optimizer=optimizer)
        
    @staticmethod
    def conv2d(layer_input, filters, f_size=4, normalization=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)

        return d

    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])

        return u
        
    # Next, we build the generator code, which uses the residual skip connections as we
    # described earlier. This is a so-called “U-Net” architecture, which is simpler to write
    # than the ResNet architecture, which some implementations use.
    def build_generator(self):
        """U-net Generator"""
        # Image input
        d0 = Input(shape=self.img_shape)
        
        # Downsampling
        d1 = self.conv2d(d0, self.gf)
        d2 = self.conv2d(d1, self.gf * 2)
        d3 = self.conv2d(d2, self.gf * 4)
        d4 = self.conv2d(d3, self.gf * 8)
        
        # Upsampling
        u1 = self.deconv2d(d4, d3, self.gf * 4)
        u2 = self.deconv2d(u1, d2, self.gf * 2)
        u3 = self.deconv2d(u2, d1, self.gf)
        
        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        
        return Model(d0, output_img)
        
    def build_discriminator(self):
        img = Input(shape=self.img_shape)
        
        d1 = self.conv2d(img, self.df, normalization=False)
        d2 = self.conv2d(d1, self.df * 2)
        d3 = self.conv2d(d2, self.df * 4)
        d4 = self.conv2d(d3, self.df * 8)
        
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same',activation='sigmoid')(d4)
        
        return Model(img, validity)
    
    def sample_images(self, epoch, batch_i):
        r, c = 2, 3

        imgs_a = self.data_loader.load_data(domaineA=True ,domaineB=False, batch_size=1)
        imgs_b = self.data_loader.load_data(domaineA=False,domaineB=True, batch_size=1)
        
        # Translate images to the other domain
        fake_b = self.g_ab.predict(imgs_a)
        fake_a = self.g_ba.predict(imgs_b)
        # Translate back to original domain
        reconstr_a = self.g_ba.predict(fake_b)
        reconstr_b = self.g_ab.predict(fake_a)

        gen_imgs = np.concatenate([imgs_a, fake_b, reconstr_a, imgs_b, fake_a, reconstr_b])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt].reshape(128,128), cmap="gray")
                #axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./%d.png" % (epoch))
        plt.show()
    
    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()
        
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        for epoch in range(epochs):
            for batch_i, (imgs_a, imgs_b) in enumerate(self.data_loader.load_batch(batch_size)):
                # ----------------------
                #  Train Discriminators
                # ----------------------
                
                # Translate images to opposite domain
                fake_b = self.g_ab.predict(imgs_a)
                fake_a = self.g_ba.predict(imgs_b)
                
                # Train the discriminators (original images = real / translated = Fake)
                da_loss_real = self.d_a.train_on_batch(imgs_a, valid)
                da_loss_fake = self.d_a.train_on_batch(fake_a, fake)
                da_loss = 0.5 * np.add(da_loss_real, da_loss_fake)

                db_loss_real = self.d_b.train_on_batch(imgs_b, valid)
                db_loss_fake = self.d_b.train_on_batch(fake_b, fake)
                db_loss = 0.5 * np.add(db_loss_real, db_loss_fake)

                # Total discriminator loss
                d_loss = 0.5 * np.add(da_loss, db_loss)
                
                # ------------------
                #  Train Generators
                # ------------------
                g_loss = self.combined.train_on_batch([imgs_a, imgs_b], [valid, valid, imgs_a, imgs_b, imgs_a, imgs_b])

                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

gan = CycleGAN()
gan.train(epochs=100, batch_size=64, sample_interval=10)