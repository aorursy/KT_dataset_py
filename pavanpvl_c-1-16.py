import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU,Flatten

#from keras.layers.convolutional_recurrent import ConvLSTM2D

from tensorflow.keras.models import Model,model_from_json

from tensorflow.keras.callbacks import TensorBoard, Callback, ModelCheckpoint

from tensorflow.keras.utils import plot_model

import scipy.io as sio 

import numpy as np

import math

import time

import keras.backend.tensorflow_backend as tfback
#%load_ext tensorboard

#%tensorboard --logdir /kaggle/working/result/TensorBoard_CsiNet_indoor_dim256_06_17_16
envir = 'indoor' #'indoor' or 'outdoor'

def _get_available_gpus():

    """Get a list of available gpu devices (formatted as strings).



    # Returns

        A list of available GPU devices.

    """

    #global _LOCAL_DEVICES

    if tfback._LOCAL_DEVICES is None:

        devices = tf.config.list_logical_devices()

        tfback._LOCAL_DEVICES = [x.name for x in devices]

    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]



tfback._get_available_gpus = _get_available_gpus
# image params

img_height = 32

img_width = 16

img_channels = 2 

img_total = img_height*img_width*img_channels

# network params

residual_num = 2

encoded_dim = 256  #compress rate=1/4->dim.=256, compress rate=1/16->dim.=64, compress rate=1/32->dim.=32, compress rate=1/64->dim.=16
# Bulid the autoencoder model of CsiNet

def residual_network(x, residual_num, encoded_dim):

    def add_common_layers(y):

        y = BatchNormalization()(y)

        y = LeakyReLU()(y)

        return y

    def residual_block_decoded(y):

        shortcut = y

        y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)

        y = add_common_layers(y)

        

        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)

        y = add_common_layers(y)

        

        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)

        y = BatchNormalization()(y)



        y = add([shortcut, y])

        y = LeakyReLU()(y)



        return y

    

    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)

    x = add_common_layers(x)

    

    

    x = Reshape((img_total,))(x)

    encoded = Dense(encoded_dim, activation='linear')(x)

    

    x = Dense(img_total, activation='linear')(encoded)

    x = Reshape((img_channels, img_height, img_width,))(x)

    for i in range(residual_num):

        x = residual_block_decoded(x)

    

    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)



    return x


image_tensor = Input(shape=(img_channels, img_height, img_width))

network_output = residual_network(image_tensor, residual_num, encoded_dim)

autoencoder = Model(inputs=[image_tensor], outputs=[network_output])

autoencoder.compile(optimizer='adam', loss='mse')

print(autoencoder.summary())
# Data loading

if envir == 'indoor':

    mat = sio.loadmat('/kaggle/input/cisnet/csi/data/DATA_Htrainin16.mat') 

    x_train = mat['HT'] # array

    mat = sio.loadmat('/kaggle/input/cisnet/csi/data/DATA_Hvalin16.mat')

    x_val = mat['HT'] # array

    mat = sio.loadmat('/kaggle/input/cisnet/csi/data/DATA_Htestin16.mat')

    x_test = mat['HT'] # array



elif envir == 'outdoor':

    mat = sio.loadmat('data/DATA_Htrainout.mat') 

    x_train = mat['HT'] # array

    mat = sio.loadmat('data/DATA_Hvalout.mat')

    x_val = mat['HT'] # array

    mat = sio.loadmat('data/DATA_Htestout.mat')

    x_test = mat['HT'] # array
x_train = x_train.astype('float32')

x_val = x_val.astype('float32')

x_test = x_test.astype('float32')

x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

"""

class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses_train = []

        self.losses_val = []



    def on_batch_end(self, batch, logs={}):

        self.losses_train.append(logs.get('loss'))

        

    def on_epoch_end(self, epoch, logs={}):

        self.losses_val.append(logs.get('val_loss'))

        



history = LossHistory()

"""

file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+time.strftime('_%m_%d_%H')+'_16'

#path = '/kaggle/working/TensorBoard_%s' %file




# load json and create model

outfile = "/kaggle/input/kernel127ecc725c/model_CsiNet_indoor_dim256_06_18_16.json"

json_file = open(outfile, 'r')

loaded_model_json = json_file.read()

json_file.close()

autoencoder = model_from_json(loaded_model_json)

# load weights outto new model

outfile = "/kaggle/input/kernel127ecc725c/model_CsiNet_indoor_dim256_06_18_16.h5"

autoencoder.load_weights(outfile)

opt = keras.optimizers.Adam(learning_rate=0.001)

autoencoder.compile(optimizer=opt, loss='mse')

"""

autoencoder.fit(x_train, x_train,

                epochs=1,

                batch_size=200,

                shuffle=True,

                validation_data=(x_val, x_val),

                callbacks=[history,

                           TensorBoard(log_dir = path)])

"""

history=autoencoder.fit(x_train, x_train,

                epochs=1000,

                batch_size=200,

                shuffle=True,

                validation_data=(x_val, x_val))
import matplotlib.pyplot as plt

plt.plot(history.history['loss'],label='train')

plt.plot(history.history['val_loss'],label='val')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend()

N_g="/kaggle/working/lossgr"+time.strftime('_%m_%d_%H')+".png"

plt.savefig(N_g)

plt.show()

#print(history.history.keys())
"""

import matplotlib.pyplot as plt

import numpy as np

a=np.array([5,4,3,2,1])

b=np.array([1,2,3,4,5])

plt.plot(a,label='v')

plt.plot(b,label='t')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

#plt.savefig('/kaggle/working/lossgr.png')

plt.legend

plt.show()

"""
if envir == 'indoor':

    mat = sio.loadmat('/kaggle/input/csifinall/DATA_HtestFin_all.mat')

    X_test = mat['HF_all']# array

#Testing data

tStart = time.time()

x_hat = autoencoder.predict(x_test)

tEnd = time.time()

print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))



# Calcaulating the NMSE and rho

"""

if envir == 'indoor':

    mat = sio.loadmat('/kaggle/input/csifinall/DATA_HtestFin_all.mat')

    X_test = mat['HF_all']# array



elif envir == 'outdoor':

    mat = sio.loadmat('data/DATA_HtestFout_all.mat')

    X_test = mat['HF_all']# array

"""

X_test = np.reshape(X_test, (len(X_test), img_height, 125))

x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))

x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))

x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)

x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))

x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))

x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))

X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)

X_hat = X_hat[:, :, 0:125]



n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))

n1 = n1.astype('float64')

n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))

n2 = n2.astype('float64')

aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))

rho = np.mean(aa/(n1*n2), axis=1)

X_hat = np.reshape(X_hat, (len(X_hat), -1))

X_test = np.reshape(X_test, (len(X_test), -1))

power = np.sum(abs(x_test_C)**2, axis=1)

power_d = np.sum(abs(X_hat)**2, axis=1)

mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

print("In "+envir+" environment")

print("When dimension is", encoded_dim)

print("NMSE is ", 10*math.log10(np.mean(mse/power)))

print("Correlation is ", np.mean(rho))
import matplotlib.pyplot as plt

'''abs'''

n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # display origoutal

    ax = plt.subplot(2, n, i + 1 )

    x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))

    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    ax.invert_yaxis()

    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 

                          + 1j*(x_hat[i, 1, :, :]-0.5))

    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    ax.invert_yaxis()

plt.show()
# save

# serialize model to JSON

model_json = autoencoder.to_json()

outfile = "/kaggle/working/model_%s.json"%file

with open(outfile, "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

outfile = "/kaggle/working/model_%s.h5"%file

autoencoder.save_weights(outfile)