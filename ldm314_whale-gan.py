import os
import gc
import pickle
import numpy as np
from pathlib import Path
#force to a gpu
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(os.listdir("../input"))
#config
INPUT_SHAPE = (384,384,1)

input_path = Path("../input/humpback-whale-identification")
train_dir = input_path / "train"
test_dir = input_path / "test"
train_csv = input_path / "train.csv"
submission_file = input_path / "sample_submission.csv"
# Read the dataset description
from pandas import read_csv

tagged = dict([(p,w) for _,p,w in read_csv(str(train_csv)).to_records()])
submit = [p for _,p,_ in read_csv(str(submission_file)).to_records()]
join   = list(tagged.keys()) + submit
len(tagged),len(submit),len(join),list(tagged.items())[:5],submit[:5]
# Determise the size of each image
from os.path import isfile
from PIL import Image as pil_image
from tqdm import tqdm

def expand_path(p):
    if isfile(str(train_dir / p)): return str(train_dir / p)
    if isfile(str(test_dir / p)): return str(test_dir / p)
    return p

p2size = {}
for p in tqdm(join):
    size      = pil_image.open(expand_path(p)).size
    p2size[p] = size
len(p2size), list(p2size.items())[:5]
# Show an example of a duplicate image (from training of test set)
import matplotlib.pyplot as plt

def show_whale(imgs, per_row=2):
    n         = len(imgs)
    rows      = (n + per_row - 1)//per_row
    cols      = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(24//per_row*cols,24//per_row*rows))
    for ax in axes.flatten(): ax.axis('off')
    for i,(img,ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))


# TODO: find if current challenge has any
# with open('../input/humpback-whale-identification-model-files/rotate.txt', 'rt') as f: rotate = f.read().split('\n')[:-1]
# rotate = set(rotate)
# rotate
def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    #if p in rotate: img = img.rotate(180)
    return img

# p    = list(rotate)[0]
# imgs = [pil_image.open(expand_path(p)), read_raw_image(p)]
# show_whale(imgs)
# Read the bounding box data from the bounding box kernel (see reference above)
with open('../input/whale-competition-bounding-boxes/bounding-box.pickle', 'rb') as f:
    p2bb = pickle.load(f)
list(p2bb.items())[:5]
# Suppress annoying stderr output when importing keras.
import sys
import platform
old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')
import keras
sys.stderr = old_stderr

import random
from keras import backend as K
from keras.preprocessing.image import img_to_array,array_to_img
from scipy.ndimage import affine_transform

img_shape    = INPUT_SHAPE # The image shape used by the model
anisotropy   = 2.15 # The horizontal compression ratio
crop_margin  = 0.05 # The margin added around the bounding box to compensate for bounding box inaccuracy

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation        = np.deg2rad(rotation)
    shear           = np.deg2rad(shear)
    rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix    = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix     = np.array([[1.0/height_zoom, 0, 0], [0, 1.0/width_zoom, 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

def read_cropped_image(p, augment):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    size_x,size_y = p2size[p]
    
    # Determine the region of the original image we want to capture based on the bounding box.
    x0,y0,x1,y1   = p2bb[p]
    #if p in rotate: x0, y0, x1, y1 = size_x - x1, size_y - y1, size_x - x0, size_y - y0
    dx            = x1 - x0
    dy            = y1 - y0
    x0           -= dx*crop_margin
    x1           += dx*crop_margin + 1
    y0           -= dy*crop_margin
    y1           += dy*crop_margin + 1
    if (x0 < 0     ): x0 = 0
    if (x1 > size_x): x1 = size_x
    if (y0 < 0     ): y0 = 0
    if (y1 > size_y): y1 = size_y
    dx            = x1 - x0
    dy            = y1 - y0
    if dx > dy*anisotropy:
        dy  = 0.5*(dx/anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx  = 0.5*(dy*anisotropy - dx)
        x0 -= dx
        x1 += dx

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5*img_shape[0]], [0, 1, -0.5*img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0)/img_shape[0], 0, 0], [0, (x1 - x0)/img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05*(y1 - y0), 0.05*(y1 - y0)),
            random.uniform(-0.05*(x1 - x0), 0.05*(x1 - x0))
            ), trans)
    trans = np.dot(np.array([[1, 0, 0.5*(y1 + y0)], [0, 1, 0.5*(x1 + x0)], [0, 0, 1]]), trans)

    # Read the image, transform to black and white and comvert to numpy array
    img   = read_raw_image(p).convert('L')
    img   = img_to_array(img)
    
    # Apply affine transformation
    matrix = trans[:2,:2]
    offset = trans[:2,2]
    img    = img.reshape(img.shape[:-1])
    img    = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant', cval=np.average(img))
    img    = img.reshape(img_shape)

    # Normalize to zero mean and unit variance
    #img  -= np.mean(img, keepdims=True)
    #img  /= np.std(img, keepdims=True) + K.epsilon()
    #normalize to -1,1
    img /= 127.5
    img -= 1.
    return img

def read_for_training(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image(p, True)

def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image(p, False)

p = list(tagged.keys())[32]
imgs = [
    read_raw_image(p),
    array_to_img(read_for_validation(p)),
    array_to_img(read_for_training(p))
]
show_whale(imgs, per_row=3)

np.max(read_for_validation(p))
from keras.layers import *
from keras.optimizers import *
from keras.models import *

from keras.utils.data_utils import *

#data generator
import random
class ImageGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_file_list, batch_size):
        self.image_file_list = np.array(image_file_list)
        self.samples = self.image_file_list.shape[0]
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X = self.__data_generation(indexes)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.samples)

    def __data_generation(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_data = np.empty((self.batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), dtype=np.float32)

        # Generate data
        for j, idx in enumerate(indices):
            img_name = self.image_file_list[idx]
            batch_data[j] = read_for_validation(img_name)

        return batch_data

def gaussian(x, mu, sigma):
    return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))


def make_2dkernel(size,sigma=1.0):
    kernel_size = size
    mean = np.floor(0.5 * kernel_size)
    kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
    # make 2D kernel
    np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=K.floatx())
    # normalize kernel by sum of elements
    kernel = np_kernel / np.sum(np_kernel)
    return kernel    

def blur_init(shape, dtype=None):
    return make_2dkernel(shape[0]).reshape(shape)
class GAN():
    def __init__(self):
        self.img_rows = INPUT_SHAPE[0]
        self.img_cols = INPUT_SHAPE[1]
        self.channels = INPUT_SHAPE[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 4250
        
        self.cleanup_memory()
        self.build_models()
        self.discriminator.summary()
        self.generator.summary()
        self.combined.summary()
    
    def build_models(self):
        self.desc_optimizer = Adam(lr=1e-6)
        self.gen_optimizer = SGD(lr=0.01, clipvalue=0.5)

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        # build the combined model
        self.discriminator.trainable = False
        self.combined = self.build_combined()

        
    def save(self,path):
        model_name = path+'-generator.h5'
        self.generator.save(model_name)
        self.generator.save_weights(model_name+".weights")

        model_name = path+'-discriminator.h5'
        self.discriminator.save(model_name)
        self.discriminator.save_weights(model_name+".weights")

#         model_name = path+'-combined.h5'
#         self.combined.save(model_name)
#         self.combined.save_weights(model_name+".weights")

    def load(self,path):
        model_name = path+'-generator.h5'
        self.generator.load_weights(model_name+".weights")

        model_name = path+'-discriminator.h5'
        self.discriminator.load_weights(model_name+".weights")

#         model_name = path+'-combined.h5'
#         self.combined.load_weights(model_name+".weights")
        
    def set_lr(self,lr):
        lr = lr #do nothing
        #K.set_value(self.discriminator.optimizer.lr, lr)
#         K.set_value(self.combined.optimizer.lr, lr)

    def build_generator(self):
        noise_input = Input(shape=(self.latent_dim,))
        x = Dense(self.latent_dim,activation='relu')(noise_input)
        x = Dense(np.prod(self.img_shape)//16,activation='relu')(x)
        x = Reshape((img_shape[0]//4,img_shape[1]//4,img_shape[2]))(x)
        x = Conv2D(16,kernel_size=(9,9),activation='tanh',padding='same')(x)
        x = Conv2D(16,kernel_size=(9,9),dilation_rate=2,activation='tanh',padding='same')(x)
        x = UpSampling2D()(x)
        x = Conv2D(8,kernel_size=(5,5),activation='tanh',padding='same')(x)
        x = Conv2D(8,kernel_size=(3,3),activation='tanh',padding='same')(x)
        x = Conv2DTranspose(8,kernel_size=(3,3),activation='tanh',strides=2,padding='same')(x)
        x = Conv2D(1,kernel_size=(3,3),activation='tanh',padding='same')(x)
        x = Conv2D(1,kernel_size=(5,5),kernel_initializer=blur_init,activation='softsign',padding='same',name='blur')(x)

        m = Model(noise_input, x, name='SGW_generator_1')
        # Freezing these keeps it a blur layer. Otherwise training changes it...        
        m.get_layer('blur').trainable=False
        return m
    
    def build_discriminator(self):
        img_input = Input(shape=self.img_shape)
        x = Conv2D(img_shape[2]*2,kernel_size=(3,3),strides=2)(img_input)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        x = Conv2D(img_shape[2]*4,kernel_size=(3,3))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        x = Conv2D(img_shape[2]*8,kernel_size=(3,3))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        x = Conv2D(img_shape[2]*16,kernel_size=(3,3))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        x = Conv2D(img_shape[2]*32,kernel_size=(3,3))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(self.latent_dim)(x)
        x = Activation('relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        m = Model(img_input, x, name='SGW_discriminator_1')
        return m

    def build_combined(self):
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        return Model(z, validity)
            
    def cleanup_memory(self):
        sess = K.get_session()
        K.clear_session()
        try:
            del self.combined
            del self.discriminator
            del self.generator
        except:
            pass
        sess.close()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        K.set_session(tf.Session(config=config))
        gc.collect()
        
    def train(self, epochs, batch_size=128, sample_interval=50):
        print("Get image list")
        img_loader = ImageGenerator(np.array(list(tagged.keys())),batch_size)

        enqueuer = OrderedEnqueuer(img_loader)
        enqueuer.start(workers=24)
        datas = enqueuer.get()

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        lr = 1e-4
        drop_interval = 5000
        drop_factor = 0.5
        for epoch in range(epochs):
            self.cleanup_memory()
            self.build_models()
            try:
                self.load('whale-gan-1-checkpoint')
            except:
                print("Checkpoint didn't load")
            
            batches_per_epoch = img_loader.samples // batch_size

            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.discriminator.trainable = True
            self.discriminator.compile(loss='binary_crossentropy',
                optimizer=self.desc_optimizer,
                metrics=['accuracy'])
            
            d_loss_samples = []
            d_acc_samples = []
            pbar = tqdm(range(batches_per_epoch))
            for i in pbar:
                imgs = next(datas)
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss_samples.append(d_loss[0])
                d_acc_samples.append(d_loss[1])
                pbar.set_description("[Disc. loss: %f, acc.: %.2f%%]" % (np.average(d_loss_samples),100*np.average(d_acc_samples)))

            attempts = 0    
            while 100*np.average(d_acc_samples) < 70.0 and attempts < 3:
                attempts+=1
                print("Discriminator accuracy too low. Continue training")
                d_loss_samples = []
                d_acc_samples = []
                pbar = tqdm(range(batches_per_epoch))
                for i in pbar:
                    imgs = next(datas)
                    noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                    gen_imgs = self.generator.predict(noise)

                    # Train the discriminator
                    d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    d_loss_samples.append(d_loss[0])
                    d_acc_samples.append(d_loss[1])
                    pbar.set_description("[Disc. loss: %f, acc.: %.2f%%]" % (np.average(d_loss_samples),100*np.average(d_acc_samples)))                

            # ---------------------
            #  Train Generator
            # ---------------------
            self.discriminator.trainable = False
            #self.combined = self.build_combined()
            self.combined.compile(loss='binary_crossentropy', optimizer=self.gen_optimizer)

            g_loss_samples = []
            pbar = tqdm(range(batches_per_epoch))
            for i in pbar:
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                g_loss = self.combined.train_on_batch(noise, valid)
                g_loss_samples.append(g_loss)
                pbar.set_description("[G loss: %f]" % (np.average(g_loss_samples)))


            attempts = 0    
            while np.average(d_loss_samples) < np.average(g_loss_samples) and attempts < 3:
                attempts+=1
                g_loss_samples = []
                lr = K.get_value(self.combined.optimizer.lr) * 0.667
                K.set_value(self.combined.optimizer.lr, lr)
                print("Discriminator winning, continue to train generator with new lr: %0.8f" % lr)
                g_loss_samples = []
                pbar = tqdm(range(batches_per_epoch))
                for i in pbar:
                    noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                    g_loss = self.combined.train_on_batch(noise, valid)
                    g_loss_samples.append(g_loss)
                    pbar.set_description("[G loss: %f]" % (np.average(g_loss_samples)))

                

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, 
                                                                   np.average(d_loss_samples), 
                                                                   100*np.average(d_acc_samples), 
                                                                   np.average(g_loss_samples) ))
            if (epoch+1) % drop_interval == 0:
                lr = lr * drop_factor
                self.set_lr(lr)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            try:
                self.save('whale-gan-1-checkpoint')
            except:
                print("Checkpoint didn't save")
        enqueuer.stop()


    def sample_images(self, epoch):
        r, c = 3, 3
        sample_noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(sample_noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        
        fig, axs = plt.subplots(r, c,figsize=(10,10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        #fig.savefig("images/%d.png" % epoch)
        plt.show()
        plt.close()
    
gc.collect()
keras.backend.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

gan = GAN()
gan.train(epochs=20, batch_size=128, sample_interval=1)
