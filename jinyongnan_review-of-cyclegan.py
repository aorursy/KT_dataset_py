import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# store all filenames
files = {}
for dirname, _, filenames in os.walk('/kaggle/input'):
    if len(filenames)==0:
        continue
    files[dirname]=[]
    for filename in filenames:
        files[dirname].append(filename)
# directories
train_horse_dir = '/kaggle/input/horse2zebra/horse2zebra/trainA'
train_zebra_dir = '/kaggle/input/horse2zebra/horse2zebra/trainB'
test_horse_dir = '/kaggle/input/horse2zebra/horse2zebra/testA'
test_zebra_dir = '/kaggle/input/horse2zebra/horse2zebra/testB'
# Imports we need in this project
!pip install git+https://www.github.com/keras-team/keras-contrib.git > /dev/null
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
# ploting
import matplotlib.pyplot as plt
# some other
from random import random
# generator a resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g
# define the standalone generator model
def define_generator(image_shape, n_resnet=1):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # c7s1-64
    g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    # u128
    g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # u64
    g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c7s1-3
    g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model
#define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    # define model
    model = Model(in_image, patch_out)
    # compile model
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model
# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(gen_model_AtoB, dis_model_B, gen_model_BtoA, image_shape):
    # ensure the model we're updating is trainable
    gen_model_AtoB.trainable = True
    # mark discriminator as not trainable
    dis_model_B.trainable = False
    # mark other generator model as not trainable
    gen_model_BtoA.trainable = False
    # discriminator element
    input_of_A = Input(shape=image_shape)
    output_of_generated_B = gen_model_AtoB(input_of_A)
    output_of_discriminated_generated_B = dis_model_B(output_of_generated_B)
    # identity element
    input_of_B = Input(shape=image_shape)
    output_of_B_generated_by_B = gen_model_AtoB(input_of_B)
    # forward cycle
    output_of_A_generated_by_generated_B = gen_model_BtoA(output_of_generated_B)
    # backward cycle
    output_of_generated_A = gen_model_BtoA(input_of_B)
    output_of_B_generated_by_generated_A = gen_model_AtoB(output_of_generated_A)
    # define model graph
    model = Model([input_of_A, input_of_B], 
                  [
                      output_of_discriminated_generated_B, 
                      output_of_B_generated_by_B, 
                      output_of_A_generated_by_generated_B, 
                      output_of_B_generated_by_generated_A
                  ]
                 )
    # define optimization algorithm configuration
    opt = Adam(lr=0.0002, beta_1=0.5)
    # compile model with mae, except discriminator result with mse
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model
# load all images in a directory into memory
def load_images(path, size=(256,256)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in files[path]:
        # load and resize the image
        pixels = load_img(path +'/'+ filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # store
        data_list.append(pixels)
    return np.asarray(data_list)

images_horse_train = load_images(train_horse_dir)
images_horse_test = load_images(test_horse_dir)
images_zebra_train = load_images(train_zebra_dir)
images_zebra_test = load_images(test_zebra_dir)
print('Shape of train horse images ',images_horse_train.shape)
print('Shape of test horse images ',images_horse_test.shape)
print('Shape of train zebra images ',images_zebra_train.shape)
print('Shape of test zebra images ',images_zebra_test.shape)
# format the images to [-1,1]
def format_images(images):
    return (images-127.5)/127.5

# format to plotable images
def format_to_images(data):
    return (data+1)/2

# select a batch of random samples, returns images and targets
def generate_real_samples(dataset, n_samples, patch_shape):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
    # generate fake instance
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y
# get formated images for training
formated_train_horse = format_images(images_horse_train)
formated_test_horse = format_images(images_horse_test)
formated_train_zebra = format_images(images_zebra_train)
formated_test_zebra = format_images(images_zebra_test)
# save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
    # save the first generator model
    filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
    g_model_AtoB.save(filename1)
    # save the second generator model
    filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
    
# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)
# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, trainA, trainB):
    # define properties of the training run
    n_epochs, n_batch, = 30, 1
    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    # prepare image pool for fake images
    poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    batch_per_epoch = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = batch_per_epoch * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        # update fakes from pool
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        # update generator B->A via adversarial and cycle loss
        g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        # update generator A->B via adversarial and cycle loss
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        # summarize performance
        # dA_loss1: A Discriminator loss of judgement in determine realA is realA
        # dA_loss2: A Discriminator loss of judgement in determine fakeA is fakeA
        # dB_loss1: B Discriminator loss of judgement in determine realB is realB
        # dB_loss2: B Discriminator loss of judgement in determine fakeB is fakeB
        # g_loss1:  BtoA Generator's loss when judged by A Discriminator
        # g_loss2:  AtoB Generator's loss when judged by B Discriminator
        #print losses every 200 steps
        if (i+1) % 5000 == 0:
            print('>Step %d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
        # save every 5 epochs
        if (i+1) % (batch_per_epoch * 5) == 0:
            # save the models
            save_models(i, g_model_AtoB, g_model_BtoA)
# shape of images (currently (256,256,3))
image_shape = formated_train_horse.shape[1:]
# generator: A -> B
# in version 3, we trained 20 epoches, let's load the weights 
g_model_AtoB = define_generator(image_shape)
g_model_AtoB.load_weights('/kaggle/input/review-of-cyclegan/g_model_AtoB_032010.h5')
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
g_model_BtoA.load_weights('/kaggle/input/review-of-cyclegan/g_model_BtoA_032010.h5')
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

print('model loaded')
# train models
# in version 3 we trained for 20 epoches
# in this version we will train 30 more epoches
# the models are supposed to have 50 epoches learning at this point
# total 50 epoches trained in version 5
# train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, formated_train_horse, formated_train_zebra)
# Generate Zebras by Horses

# select 10 horses
img_indexes = np.random.choice(len(formated_test_horse),10)
selected_horses = formated_test_horse[img_indexes]
generated_zebras = g_model_AtoB.predict(selected_horses)
fig = plt.figure(figsize=(6,30))
for i in range(len(selected_horses)):
    # re-format to images plotable
    plotable_horses = format_to_images(selected_horses)
    plotable_zebras = format_to_images(generated_zebras)
    # plot horses
    sp = plt.subplot(10,2,i*2+1)
    sp.set_title('horse'+str(i+1))
    sp.axis('off')
    plt.imshow(plotable_horses[i])
    # plot zebras(generated)
    sp = plt.subplot(10,2,i*2+2)
    sp.set_title('zebra'+str(i+1)+'(generated)')
    sp.axis('off')
    plt.imshow(plotable_zebras[i])

# Generate Horses by Zebras

# select 10 zebras
img_indexes = np.random.choice(len(formated_test_zebra),10)
selected_zebras = formated_test_zebra[img_indexes]
generated_horses = g_model_BtoA.predict(selected_zebras)
fig = plt.figure(figsize=(6,30))
for i in range(len(selected_zebras)):
    # re-format to images plotable
    plotable_zebras = format_to_images(selected_zebras)
    plotable_horses = format_to_images(generated_horses)
    # plot zebras
    sp = plt.subplot(10,2,i*2+1)
    sp.set_title('zebra'+str(i+1))
    sp.axis('off')
    plt.imshow(plotable_zebras[i])
    # plot horses(generated)
    sp = plt.subplot(10,2,i*2+2)
    sp.set_title('horse'+str(i+1)+'(generated)')
    sp.axis('off')
    plt.imshow(plotable_horses[i])
# save models
! cp '/kaggle/input/review-of-cyclegan/g_model_AtoB_032010.h5' '/kaggle/working/g_model_AtoB_032010.h5'
! cp '/kaggle/input/review-of-cyclegan/g_model_BtoA_032010.h5' '/kaggle/working/g_model_BtoA_032010.h5'