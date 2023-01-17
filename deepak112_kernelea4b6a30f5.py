import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from keras.datasets import mnist
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import initializers
import os
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
from skimage.transform import rescale, resize
from keras.layers.convolutional import Conv2D, Conv2DTranspose
print(os.listdir("../"))


# Any results you write to the current directory are saved as output.
print(os.mkdir("../output"))
print(os.listdir("../"))

def load_path(path):
    directories = []
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories
np.random.seed(10)
def load_data(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                files.append(data.imread(os.path.join(d,f)))
                file_names.append(os.path.join(d,f))
                count = count + 1
    return files
            
files = load_data(load_path("../input/anime-dataset/faces/"), ".jpg")



def res_block(model, kernal_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(0.2)(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
        
    model = keras.layers.add([gen, model])
    
    return model
    
    
def block(model, kernal_size, filters, strides):
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    #shape = [model.shape[0].value, model.shape[1].value, model.shape[2].value, model.shape[3].value]
    #model = Utills.SubpixelConv2D(shape, 2)(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


class Generator(object):

    def __init__(self, noise_shape):
        self.noise_shape = noise_shape

    def generator(self):
	    gen_input = Input(shape = self.noise_shape)
	    
	    model = Conv2DTranspose(filters = 512, kernel_size = 4, strides = 1, padding = "same")(gen_input)
	    model = BatchNormalization(momentum = 0.5)(model)
	    model = LeakyReLU(alpha = 0.2)(model)
	    
	    model = Conv2DTranspose(filters = 256, kernel_size = 4, strides = 2, padding = "same")(model)
	    model = BatchNormalization(momentum = 0.5)(model)
	    model = LeakyReLU(alpha = 0.2)(model)
	    
	    model = Conv2DTranspose(filters = 128, kernel_size = 4, strides = 2, padding = "same")(model)
	    model = BatchNormalization(momentum = 0.5)(model)
	    model = LeakyReLU(alpha = 0.2)(model)
	    
	    model = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same")(model)
	    model = BatchNormalization(momentum = 0.5)(model)
	    model = LeakyReLU(alpha = 0.2)(model)
	    
	    
	    model = Conv2DTranspose(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
	    model = BatchNormalization(momentum = 0.5)(model)
	    model = LeakyReLU(alpha = 0.2)(model)
	    
	    model = Conv2DTranspose(filters = 3, kernel_size = 4, strides = 2, padding = "same")(model)
	    
	    model = Activation('tanh')(model)
	    
	    generator_model = Model(inputs = gen_input, outputs = model)
	    
	    return generator_model


class Discriminator(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape
    
    def discriminator(self):
        dis_input = Input(shape = self.image_shape)
        
        model = discriminator_block(dis_input, 64, 4, 2)
        
        model = discriminator_block(model, 128, 4, 2)
        
        model = discriminator_block(model, 256, 4, 2)

        model = discriminator_block(model, 512, 4, 2)
       
        model = Flatten()(model)
        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        generator_model = Model(inputs = dis_input, outputs = model)
        
        return generator_model
    
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],input_shape[1] * scale,input_shape[2] * scale,int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape
    
    def subpixel(x):
        return tf.depth_to_space(x, scale)
        
    return Lambda(subpixel, output_shape=subpixel_shape)
    
def load_path(path):
    directories = []
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories
    
def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                files.append(data.imread(os.path.join(d,f)))
                file_names.append(os.path.join(d,f))
                count = count + 1
    return files     
            
def load_data_from_dirs_resize(dirs, ext, size):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                files.append(resize(data.imread(os.path.join(d,f)), size))
                file_names.append(os.path.join(d,f))
                count = count + 1
    return files     
                        
          
def load_data(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext)
    return files
    
def load_data_as_array(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext)
    files = array(files)
    return files
    
def load_data_resize(directory, ext, size):

    files = load_data_from_dirs_resize(load_path(directory), ext, size)
    return files
    
def load_data_as_array_resize(directory, ext, size):

    files = load_data_from_dirs_resize(load_path(directory), ext, size)
    files = array(files)
    return files

def normalize(input_data):

    input_data = (input_data.astype(np.float32) - 127.5)/127.5
    return input_data
    
def denormalize(input_data):

    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8) 
    
    

	
image_shape = (96,96,3)
noise_shape = (6,6,100)


x_train = files[:8000]
x_train = array(x_train)
x_train = normalize(x_train)
plt.imshow(x_train[134])
x_train.shape
Gen = Generator(noise_shape) 
generator = Gen.generator()
generator.summary()
def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def plot_generated_images(epoch, generator, examples=25, dim=(5, 5), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples,  6, 6, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 96, 96, 3)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.savefig('../output/gan_generated_image_epoch_%d.png' % epoch)

def train(epochs=4000, batch_size=16):
    
    batch_count = int(x_train.shape[0] / batch_size)
    
    adam = Adam(lr=0.0002, beta_1=0.5)
    
    Dis = Discriminator(image_shape)
    discriminator = Dis.discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    Gen = Generator(noise_shape) 
    generator = Gen.generator()
    generator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    gan = get_gan_network(discriminator, noise_shape, generator, adam)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in range(batch_count):
            noise = np.random.normal(0, 1, size=[batch_size, 6, 6, 100])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            generated_images = generator.predict(noise)
            #X = np.concatenate([image_batch, generated_images])

            #y_dis = np.zeros(2*batch_size)
            #y_dis[:batch_size] = 1
            
            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2

            discriminator.trainable = True
            #discriminator.train_on_batch(X, y_dis)
            discriminator.train_on_batch(image_batch, real_data_Y)
            discriminator.train_on_batch(generated_images, fake_data_Y)
            

            noise = np.random.normal(0, 1, size=[batch_size,  6, 6, 100])
            #y_gen = np.ones(batch_size)
            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            #gan.train_on_batch(noise, y_gen)
            gan.train_on_batch(noise, gan_Y)
            

        if e == 1 or e % 5 == 0:
            plot_generated_images(e, generator)
        
    generator.save('../output/gen_model%d.h5' % e) 
    discriminator.save('../output/dis_model%d.h5' % e) 
    gan.save('../output/gan_model%d.h5' % e) 
    
    return generator, discriminator, gan
    

generator, discriminator, gan = train(300,64)






