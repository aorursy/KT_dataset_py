

import cv2

import numpy as np # linear algebra

import keras

from keras import backend as K

from keras import layers

import random

import os

print(os.listdir("../input"))



train_dir = '../input/cityscapes_data/cityscapes_data/train/'

val_dir = '../input/cityscapes_data/cityscapes_data/val/'

height = 256

width = 512

channels=3



IDS_train=[]

for path in os.listdir(train_dir):

      

       IDS_train.append(train_dir + '/' + path)

IDS_val=[]

print(IDS_train[-5:])

for path in os.listdir(val_dir):

       

       IDS_val.append(val_dir + '/' + path)

print(IDS_val[-5:])

l=len(IDS_train)

partition={'train':IDS_train, 'val':IDS_val}

import matplotlib.pyplot as plt





def show_images(images):

  # images=(no of images, width, height, channels)

  n_x = np.int(np.sqrt(images.shape[0]))

  n_y = np.int(np.sqrt(images.shape[0]))

  #x and y coordinates in the images we will display

  coor_x =images.shape[1]

  coor_y =images.shape[2]

  

  figure = np.zeros((coor_x * n_x, coor_y * n_y, images.shape[3]))



  for i in range(n_x):  

    for j in range(n_y):  

      ind = i+n_x*j

      if ind >= images.shape[0]:

        break

      image = images[ind, :,:,:]

      figure[i * coor_x: (i + 1) * coor_x, j * coor_y: (j + 1) * coor_y] = image



  plt.figure(figsize=(20, 10))

  plt.imshow(np.squeeze(figure))

  ax = plt.gca()

  

  ax.grid(False)



  plt.show()

random.shuffle(IDS_val)

im= np.empty((10, 256,512, 3))



for j in range(10):

    x=cv2.imread(IDS_val[j])

   

    im[j]=x/255 #normalizing

 



show_images(im)




img_input = keras.Input(shape=(height, height, channels))

#Encoder

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(img_input)

x = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)

x = layers.MaxPooling2D((2, 2), padding='same')(x)







#Decoder

x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)

x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)













img_output = x



autoencoder = keras.models.Model(img_input, img_output)

autoencoder.summary()

autoencoder.compile(optimizer='rmsprop', loss='mse')



class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=16, dim=(256,256), n_channels=3,

                shuffle=True):

        'Initialization'

        self.dim = dim

        self.batch_size = batch_size

        self.list_IDs = list_IDs

        self.n_channels = n_channels

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        #print(list_IDs_temp)



        # Generate data

        X = self.__data_generation(list_IDs_temp)



        return X



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        

        Z = np.empty((self.batch_size, 256,512, self.n_channels))

        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            Z[i]= cv2.imread(ID)

            X[i] = np.asarray(Z[i][:,:256,:]/255)





        return X,X
# Generators

train_gen = DataGenerator(IDS_train)

val_gen = DataGenerator(IDS_val)

#We assign early stopping in case we overfit

early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=2)

history = autoencoder.fit_generator(train_gen,

                              steps_per_epoch=train_gen.__len__(),

                              epochs=5, verbose=1,

                              validation_data=val_gen, callbacks=[early_stop], validation_steps=val_gen.__len__())
#plot Loss



import matplotlib.pyplot as plt



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(loss))



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()
random.shuffle(IDS_val)

im= np.empty((32, 256,256, 3))



msk=np.empty((32, 256,256, 3))

for j in range(32):

    x=cv2.imread(IDS_val[j])

   

    im[j]=x[:,:height,:]/255 #normalizing

    msk[j]=x[:,height:,:]/255

    msk=np.asarray(msk)







ae_images = autoencoder.predict(im)

show_images(ae_images)

show_images(im)
#discriminator



discriminator_input = layers.Input(shape=(height, height, channels))





x = layers.Conv2D(128, 3)(discriminator_input)

x = layers.BatchNormalization()(x)

x = layers.LeakyReLU()(x)



x = layers.Conv2D(128, 4, strides=2)(x)

x = layers.BatchNormalization()(x)

x = layers.LeakyReLU()(x)



x = layers.Conv2D(128, 4, strides=2)(x)

x = layers.BatchNormalization()(x)

x = layers.LeakyReLU()(x)



x = layers.Conv2D(128, 4, strides=2)(x)

x = layers.BatchNormalization()(x)

x = layers.LeakyReLU()(x)



x = layers.Flatten()(x)





x = layers.Dropout(0.5)(x)





discriminator_output = layers.Dense(1, activation='sigmoid')(x)



discriminator = keras.models.Model(discriminator_input, discriminator_output)

discriminator.summary()



# To stabilize training, we use learning rate decay

# and gradient clipping (by value) in the optimizer.

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
#We define the adversarially trained autoencoder using the discriminator

#When adversarialy training the autoencoder, we do not want to update the discriminator

# (will only apply to the `gan` model)



gan_input = keras.Input(shape=(height, height, channels,))

main_output = autoencoder(gan_input)

auxiliary_output=discriminator(main_output)

discriminator.trainable = False

gan = keras.models.Model(gan_input,  outputs=[main_output, auxiliary_output])



gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)

gan.compile(optimizer=gan_optimizer,loss=['mse','binary_crossentropy'], loss_weights=[.7, .3])

gan.summary()
# Start training loop

batch_size = 32

def batch_generator(IDS):

    batch=np.empty((batch_size, height,height, channels))

    

    for j in range(len(IDS)):

        x=cv2.imread(IDS[j])

        x=x[:,:height,:]

        batch[j]=np.asarray(x)/255

    return batch

start = 0

epoch = 0

num_epochs=5

adv_loss = np.zeros((num_epochs, 3))

val_adv_loss = np.zeros((num_epochs, 3))

disc_loss = np.zeros((num_epochs, 1))

while epoch < num_epochs:

    #Step 1:train the discriminator

    #We take random images

    random.shuffle(IDS_train)

    stop=start+batch_size

    IDS_t=IDS_train[start:stop]

    real_images=batch_generator(IDS_t)





    #We take arbitrary generated images

    random.shuffle(IDS_train)

    stop=start+batch_size

    IDS_t=IDS_train[start:stop]

    generated_images = autoencoder.predict(real_images)

    

    #We combine them

    combined_images = np.concatenate([generated_images, real_images])



    # Assemble labels discriminating real from fake images

    labels = np.concatenate([np.ones((batch_size, 1)),

                             np.zeros((batch_size, 1))])  # 1=fake , 0=real

    # Add random noise to the labels

    labels += 0.05 * np.random.random(labels.shape)



    #We feed them to the discriminator to recognize real from fake images

    d_loss = discriminator.train_on_batch(combined_images, labels)



    #Step 2: Update autoencoder to fool discriminator

    # Make misleading targets 'all real images'

    misleading_targets = np.zeros((batch_size, 1))





    #training data that includes random_images with target generated images and misleading_targets

    a_loss = gan.train_on_batch(real_images, [generated_images, misleading_targets])

    

    #Computing validation loss

    #We shuffle the valuation ID's

    random.shuffle(IDS_val)

    IDS_v=IDS_val[:batch_size]

    real_images_v=batch_generator(IDS_v)

    generated_images_v=autoencoder.predict(real_images_v)



    labels_v = np.zeros((batch_size, 1))  # 1=fake, 0=real

    v_loss= gan.test_on_batch(real_images_v,[generated_images_v,labels_v])                         



    start += batch_size

    if start > len(IDS_train) - batch_size:

        start = 0

        

        # We print metrics

        print('discriminator loss at epoch %s: %s' % (epoch, d_loss))

  

        disc_loss[epoch,:] = d_loss



        print('adversarial loss at epoch %s: %s' % (epoch, a_loss))



        adv_loss[epoch,:] = a_loss

    

        val_adv_loss[epoch,:] = v_loss

        print('val_adv loss at epoch %s: %s' % (epoch, v_loss))

        epoch+=1
x_train=batch_generator(IDS_val[0:25])

ae_images, discrim = gan.predict(x_train)

show_images(ae_images)

show_images(x_train)



loss1 = adv_loss

val_loss = val_adv_loss

epochs = range(num_epochs)



plt.figure()



plt.plot(epochs, disc_loss, 'bo', label='Discriminator Training loss')

plt.plot(epochs, loss1, 'mo', label='Training loss')

plt.plot(epochs, val_loss, 'co', label='Validation loss')



plt.title('Losses')

plt.legend()



plt.show()
''' Any comments are more than welcome. Thank you for taking the time to read my kernel.'''