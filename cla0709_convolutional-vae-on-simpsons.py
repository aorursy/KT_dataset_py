import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn
from collections import Counter
import glob
import pickle
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Lambda, Dense, Dropout, Activation, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.losses import mse, binary_crossentropy
from keras.optimizers import SGD, Adam
from random import shuffle

train = False
%matplotlib inline
map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak', 
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}
pic_size = 64
batch_size = 32
epochs = 200
num_classes = len(map_characters)
 
def load_test_set(path):
    pics, labels = [], []
    reverse_dict = {v:k for k,v in map_characters.items()}
    for pic in glob.glob(path+'*.*'):
        char_name = "_".join(pic.split('/')[5].split('_')[:-1])
        if char_name in reverse_dict:
            temp = cv2.imread(pic)
            temp = cv2.resize(temp, (pic_size,pic_size)).astype('float32') / 255.
            pics.append(temp)
            labels.append(reverse_dict[char_name])
    X_test = np.array(pics)
    y_test = np.array(labels)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("Test set", X_test.shape, y_test.shape)
    return X_test, y_test
X_test, y_test = load_test_set("../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/")
#model.fit(X_test, y_test, epochs = 60)
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_VAE(input_shape):
    image_size = input_shape[1]
    original_dim = image_size * image_size
    inputs = Input(shape=input_shape)
    print(inputs.shape)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    print(x.shape)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    print(x.shape)
    x = Dropout(0.2)(x)
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    print(x.shape)
    x = Dropout(0.2)(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    print(x.shape)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    print(x.shape)
    x = Dense(128, activation='relu')(x)
    print(x.shape)
    
    latent_dim = 20
    
    latent_mean = Dense(latent_dim)(x)
    print(latent_mean.shape)
    latent_log_variance = Dense(latent_dim)(x)
    print(latent_log_variance.shape)
    
    latent_sample = Lambda(sampling)([latent_mean, latent_log_variance])
    print(latent_sample.shape)
    
    encoder = Model(inputs, [latent_mean, latent_log_variance, latent_sample])
    
    latent_inputs = Input(shape=(latent_dim,))
    print(latent_inputs.shape)
    x = Dense(8*8*256, activation='relu')(latent_inputs)
    print(x.shape)
    x = Reshape((8,8,256))(x)
    print(x.shape)
    x = UpSampling2D((2, 2))(x)
    print(x.shape)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    
    x = UpSampling2D((2, 2))(x)
    print(x.shape)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    
    x = UpSampling2D((2, 2))(x)
    print(x.shape)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    print(x.shape)
    
    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    print(outputs.shape)
    
    decoder = Model(latent_inputs, outputs)
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs)
    
    reconstruction_loss = binary_crossentropy(inputs, outputs) * original_dim
    reconstruction_loss = K.mean(reconstruction_loss)
    print(reconstruction_loss)
    kl_loss = 1 + latent_log_variance - K.square(latent_mean) - K.exp(latent_log_variance)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    print(kl_loss)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    print(vae_loss)
    vae.add_loss(vae_loss)
    return vae, encoder, decoder
vae, encoder, decoder = create_VAE(input_shape=(pic_size,pic_size,3))
vae.compile(optimizer='adam', metrics=['accuracy'])
if train:
    vae.fit(X_test, batch_size=32, epochs=100)
else:
    vae.load_weights("../input/convolutional-vae-on-simpsons/sicche.h5")
    encoder.load_weights("../input/convolutional-vae-on-simpsons/encodersicche.h5")
    decoder.load_weights("../input/convolutional-vae-on-simpsons/decodersicche.h5")
image_test = X_test[3]
image_reconstruction = vae.predict(np.expand_dims(image_test, axis = 0))[0]


F = plt.figure(1, (15,20))
grid = AxesGrid(F, 111, nrows_ncols=(1, 2), axes_pad=0, label_mode="1")
grid[0].imshow(cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB))
grid[1].imshow(cv2.cvtColor(image_reconstruction, cv2.COLOR_BGR2RGB))
vae.save_weights("sicche.h5")
encoder.save_weights("encodersicche.h5")
decoder.save_weights("decodersicche.h5")