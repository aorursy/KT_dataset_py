from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from glob import glob
import seaborn as sns
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import time
import os
from keras.preprocessing.image import img_to_array
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
def pixalate_image(image, scale_percent = 40):
    width = int(256 * 40 / 250)
    height = int(256 * 40 / 250)
    dim = (width, height)

    small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
    # scale back to original size
    width = int(small_image.shape[1] * 250 / scale_percent)
    height = int(small_image.shape[0] * 250 / scale_percent)
    dim = (width, height)

    low_res_image = cv2.resize(small_image, dim, interpolation = cv2.INTER_AREA)

    return low_res_image
def load_data(batch_size):
    
    path1=sorted(glob('../input/covidct/COVID-CT/CT_COVID/*'))#test data
    i=np.random.randint(0,27)
    #noise_factor = 0.5
   # batch_size=1
    batch1=path1[i*batch_size:(i+1)*batch_size]

    img_A=[]
    img_B=[]
    for filename1 in batch1:
        img1=cv2.imread(filename1,0)
        img1=img1[...,::-1]
        img2=pixalate_image(img1)
        img1=cv2.resize(img1,(256,256),interpolation=cv2.INTER_AREA)
        img2=cv2.resize(img2,(256,256),interpolation=cv2.INTER_AREA)
        img_A.append(img1)
        img_B.append(img2)
        
    img_A = np.reshape(img_A, (len(img_A), 256, 256, 1))
    img_B = np.reshape(img_B, (len(img_B), 256, 256, 1))
    img_A=np.array(img_A)/127.5-1
    img_B=np.array(img_B)/127.5-1

    
    return img_A,img_B 

def load_batch(batch_size):
    path1=sorted(glob('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/*'))#training data
    n_batches=int(len(path1)/batch_size)
    
    for i in range(n_batches):
        batch1=path1[i*batch_size:(i+1)*batch_size]
        img_A,img_B=[],[]
        for filename1 in batch1:
            img1=cv2.imread(filename1,0)
            img1=img1[...,::-1]
            img2=pixalate_image(img1)
            img1=cv2.resize(img1,(256,256),interpolation=cv2.INTER_AREA)    
            img2=cv2.resize(img2,(256,256),interpolation=cv2.INTER_AREA)
            img_A.append(img2)#noisy
            img_B.append(img1)#clean
        
        img_A = np.reshape(img_A, (len(img_A), 256, 256, 1))
        img_B = np.reshape(img_B, (len(img_B), 256, 256, 1))
        img_A=np.array(img_A)/127.5-1
        img_B=np.array(img_B)/127.5-1

    
        yield img_A,img_B #return generator
class pix2pix():
    def __init__(self):
        self.img_rows=256
        self.img_cols=256
        self.channels=1
        self.img_shape=(self.img_rows,self.img_cols,self.channels)
        
        # Calculate output shape of D (PatchGAN)
        patch=int(self.img_rows/(2**4))
        self.disc_patch=(patch,patch,1)
    
        # Number of filters in the first layer of G and D
        self.gf=64
        self.df=64
    
        optimizer=tf.keras.optimizers.Adam(0.0002,0.5)
    
        # Build and compile the discriminator
        self.discriminator=self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                    optimizer=optimizer,
                                    metrics=['accuracy'])
        
        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator=self.build_generator()
        #self.generator.summary()
        
        # Input images and their conditioning images
        img_A=layers.Input(shape=self.img_shape)
        img_B=layers.Input(shape=self.img_shape)

        
        img=self.generator(img_A)
        
        # For the combined model we will only train the generator
        self.discriminator.trainable=False
    
        valid=self.discriminator([img,img_A])
    
        #self.combined=Model(img_A,valid)
        #self.combined.compile(loss='binary_crossentropy',
        #                     optimizer=optimizer)
    
        self.combined =Model(inputs=[img_B, img_A], outputs=[valid, img])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
    def build_generator(self):
        def conv2d(layer_input,filters,f_size=(4,4),bn=True):
            d=layers.Conv2D(filters,kernel_size=f_size,strides=(2,2),padding='same')(layer_input)
            d=layers.LeakyReLU(0.2)(d)
            if bn:
                d=layers.BatchNormalization()(d)
            return d
    
        def deconv2d(layer_input,skip_input,filters,f_size=(4,4),dropout_rate=0):
            u=layers.UpSampling2D((2,2))(layer_input)
            u=layers.Conv2D(filters,kernel_size=f_size,strides=(1,1),padding='same',activation='relu')(u)
            if dropout_rate:
                u=layers.Dropout(dropout_rate)(u)
            u=layers.BatchNormalization()(u)
            u=layers.Concatenate()([u,skip_input])
            return u
    
        d0=layers.Input(shape=self.img_shape)
    
        d1=conv2d(d0,self.gf,bn=False) 
        d2=conv2d(d1,self.gf*2)         
        d3=conv2d(d2,self.gf*4)         
        d4=conv2d(d3,self.gf*8)         
        d5=conv2d(d4,self.gf*8)         
        d6=conv2d(d5,self.gf*8)        
    
        d7=conv2d(d6,self.gf*8)         
    
        u1=deconv2d(d7,d6,self.gf*8,dropout_rate=0.5)   
        u2=deconv2d(u1,d5,self.gf*8,dropout_rate=0.5)   
        u3=deconv2d(u2,d4,self.gf*8,dropout_rate=0.5)   
        u4=deconv2d(u3,d3,self.gf*4)   
        u5=deconv2d(u4,d2,self.gf*2)   
        u6=deconv2d(u5,d1,self.gf)     
        u7=layers.UpSampling2D((2,2))(u6)
    
        output_img=layers.Conv2D(self.channels,kernel_size=(4,4),strides=(1,1),padding='same',activation='tanh')(u7)
    
        return Model(d0,output_img)
  
    def build_discriminator(self):
        def d_layer(layer_input,filters,f_size=(4,4),bn=True):
            d=layers.Conv2D(filters,kernel_size=f_size,strides=(2,2),padding='same')(layer_input)
            d=layers.LeakyReLU(0.2)(d)
            if bn:
                d=layers.BatchNormalization()(d)
            return d
    
        img_A=layers.Input(shape=self.img_shape)
        img_B=layers.Input(shape=self.img_shape)
    
        combined_imgs=layers.Concatenate(axis=-1)([img_A,img_B])
    
        d1=d_layer(combined_imgs,self.df,bn=False)
        d2=d_layer(d1,self.df*2)
        d3=d_layer(d2,self.df*4)
        d4=d_layer(d3,self.df*8)
    
        validity=layers.Conv2D(1,kernel_size=(4,4),strides=(1,1),padding='same',activation='sigmoid')(d4)
    
        return Model([img_A,img_B],validity)

  
    def train(self,epochs,batch_size=1):
        def plot_history(d_hist,g_hist,a_hist):
            sns.set()

            # plot loss
            plt.subplot(2, 1, 1)
            plt.plot(d_hist, label='D loss')
            plt.plot(g_hist, label='G loss')
            plt.title('discriminator and generator loss')
            plt.legend()
            #plot accuracy
            plt.subplot(2, 1, 2)
            plt.plot(a_hist, label='accuracy')
            plt.title('Accuracy')
            plt.legend()

            plt.show()
            
        valid=np.ones((batch_size,)+self.disc_patch)
        fake=np.zeros((batch_size,)+self.disc_patch)
          
        for epoch in range(epochs):
            d_hist=[]
            g_hist=[]
            a_hist=[]
            start=time.time()
            for batch_i,(img_A,img_B) in enumerate(load_batch(1)):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                # Condition on B and generate a translated version
                gen_imgs=self.generator.predict(img_A)
        
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([img_B, img_A], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, img_A], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([img_A, img_B], [valid, img_B])
                

                if batch_i % 50 == 0:
                    print ("[Epoch %d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %f]" % (epoch,
                                                                                            batch_i,
                                                                                            d_loss[0],100*d_loss[1],
                                                                                            g_loss[0]))
                           
                    d_hist.append(d_loss[0])
                    g_hist.append(g_loss[0])
                    a_hist.append(d_loss[1])
                
            self.sample_images(epoch=10)
            print('Time for epoch {} is {} sec'.format(epoch,time.time()-start))
            
            plot_history(d_hist,g_hist,a_hist)

    def sample_images(self, epoch):
        r, c = 3, 3
        img_A, img_B =load_data(3)
        fake_A = self.generator.predict(img_A)

        gen_imgs = np.concatenate([img_A, fake_A, img_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt].reshape(256,256), cmap="gray")
                axs[i,j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./%d.png" % (epoch))
        plt.show()
if __name__ == '__main__':
    gan = pix2pix()
    gan.train(epochs=5, batch_size=1)