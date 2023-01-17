import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import os

import random

import tensorflow as tf



from skimage.exposure import rescale_intensity

from matplotlib.colors import rgb_to_hsv





import six

import keras

from keras import applications

from keras.models import Model, Sequential

from keras.callbacks import ModelCheckpoint

from keras.layers import (

    Add,

    Convolution2D,

    Input,

    Activation,

    Dense,

    Flatten,

    Dropout,

    LSTM,

    Reshape,

    TimeDistributed,

    ELU,

    Bidirectional

)

from keras.layers.convolutional import (

    Conv2D,

    Conv3D,

    MaxPooling2D,

    MaxPooling3D,

    AveragePooling2D,

    AveragePooling3D

)

from keras.layers.merge import add

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

from keras import backend as K

from keras.engine import InputSpec, Layer
def squash(x, axis=-1):

    # s_squared_norm is really small

    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()

    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)

    # return scale * x

    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)

    scale = K.sqrt(s_squared_norm + K.epsilon())

    return x / scale



class Capsule(Layer):

    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,

                 activation='default', **kwargs):

        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule

        self.dim_capsule = dim_capsule

        self.routings = routings

        self.kernel_size = kernel_size

        self.share_weights = share_weights

        if activation == 'default':

            self.activation = squash

        else:

            self.activation = Activation(activation)



    def build(self, input_shape):

        super(Capsule, self).build(input_shape)

        input_dim_capsule = input_shape[-1]

        if self.share_weights:

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(1, input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     # shape=self.kernel_size,

                                     initializer='glorot_uniform',

                                     trainable=True)

        else:

            input_num_capsule = input_shape[-2]

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(input_num_capsule,

                                            input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     initializer='glorot_uniform',

                                     trainable=True)



    def call(self, u_vecs):

        if self.share_weights:

            u_hat_vecs = K.conv1d(u_vecs, self.W)

        else:

            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])



        batch_size = K.shape(u_vecs)[0]

        input_num_capsule = K.shape(u_vecs)[1]

        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,

                                            self.num_capsule, self.dim_capsule))

        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]



        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]

        for i in range(self.routings):

            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]

            c = K.softmax(b)

            c = K.permute_dimensions(c, (0, 2, 1))

            b = K.permute_dimensions(b, (0, 2, 1))

            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))

            if i < self.routings - 1:

                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])



        return outputs



    def compute_output_shape(self, input_shape):

        return (None, self.num_capsule, self.dim_capsule)
BATCH_SIZE = 20

IMG_H = 180

IMG_W =  180

IMG_C = 1

keep_prob = .2

SEQ = 10

LEFT_PAD = 5
test_angles = pd.read_csv("/kaggle/input/testing-true-csv/testing_cleaned.csv").steering_angle



train_angles = pd.read_csv("/kaggle/input/training-testing-csv/cleaned.csv").angle



test_angles.head(3)
def prep(pimg, img):

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    pimg = cv2.cvtColor(pimg, cv2.COLOR_GRAY2RGB)

    # Convert to HSV format

    pimg = rgb_to_hsv(pimg)

    img = rgb_to_hsv(img)

    # Subtract images

    img = img[:, :, 2] - pimg[:, :, 2]

    # Rescale intensity

    img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))

    img = np.array(img, dtype=np.uint8)

    # Return differenced image

    return img



def construct_seq (path, angles):

    folder = os.listdir(path)

    folder.sort()

    size = len(folder)

    data = []

    for i in range (size):

        start = i-((SEQ+LEFT_PAD)-1)

        seq = []

        for j in range (start, i+1):

            j = 0 if j < 0 else j

            img = folder[j]

            seq.append(img)

        data.append([seq, angles[i]])

        print("Constructing sequences: {} of {}".format(i, size), end="\r")

    return data



def load_images (path, gray=True):

    dic = {}

    folder = os.listdir(path)

    folder.sort()

    pimg = None

    for i, file in enumerate(folder):

        img = cv2.imread(path+"/"+file, cv2.IMREAD_GRAYSCALE) if gray else cv2.imread(path+"/"+file)

        #img = img[220:, :]

        img = cv2.resize(img, (IMG_W, IMG_H))

        #pimg = img if pimg is None else pimg

        #img_r = prep(pimg, img)

        dic[file] = img

        #pimg = img

        print("Loading: {} of {}".format(i, len(folder)), end="\r")

    return dic





def load_data (path, gray=True, test=False):

    data = []

    angles = test_angles if test else train_angles

    

    folder = os.listdir(path)

    folder.sort()

    pimg = None

    for i, file in enumerate(folder):

        img = cv2.imread(path+"/"+file, cv2.IMREAD_GRAYSCALE) if gray else cv2.imread(path+"/"+file)

        #img = img[220:, :]

        img = cv2.resize(img, (IMG_W, IMG_H))

        pimg = img if pimg is None else pimg

        img_r = prep(pimg, img)

        pimg = img

        

        data.append([img, angles[i]])

        print("Loading: {} of {}".format(i, len(folder)), end="\r")

    return dic
path_train_images = "/kaggle/input/training-testing-images-pure/train_center/train_center"

path_test_images = "/kaggle/input/training-testing-images-pure/test_center/test_center"


print ("Loading training images into dictionary.")

train_images = load_images(path_train_images)



print ("Loading testing images into dictionary.")

test_images = load_images(path_test_images)



print ("Constructing training images sequences.")

train_seq = construct_seq(path_train_images, train_angles)



print ("Constructing testing images sequences.")

test_seq = construct_seq(path_test_images, test_angles)



# Shuffle training sequences to perform better training

random.shuffle(train_seq)
for i, file in enumerate(train_images):

    if (i == 1000):

        img = train_images[file]

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        plt.imshow(img)

        break


folder = os.listdir(path_train_images)

folder.sort()

img = cv2.imread(path_train_images+"/"+folder[550], cv2.IMREAD_GRAYSCALE)

img = img[180:, :]

print(img.shape)

img = cv2.resize(img, (IMG_W, IMG_H))



plt.imshow(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, sequences_ids, img_dict, batch_size=32, dim=(15, 200, 200), n_channels=1, shuffle=False):

        'Initialization'

        self.batch_size = batch_size

        self.n_channels = n_channels

        self.img_dict = img_dict

        self.dim = dim

        self.sequences_ids = sequences_ids

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.sequences_ids) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        seq_tmp = [self.sequences_ids[k] for k in indexes]



        # Generate data

        X, y = self.__data_generation(seq_tmp)



        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.sequences_ids))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, seq_tmp):

        

        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size))



        for i, pack in enumerate(seq_tmp):

            # Read angle, and sequence image names

            angle = pack[1]

            seq = pack[0]

            # Load images from image dictionary

            images = []

            for img_name in seq:

                img = self.img_dict[img_name]

                images.append(img)

            # Convert images to numpy array

            images = np.array(images).reshape(-1, *self.dim, self.n_channels)

            images = images.astype('float32')

            # Normalize images

            images /= 255.0

            #images = -1 + (2 * images)

            X[i,] = images

            y[i] = angle

        

        return X, y

train_gen = DataGenerator(train_seq, train_images, batch_size=BATCH_SIZE, dim=(SEQ+LEFT_PAD, IMG_H, IMG_W), n_channels=IMG_C)

test_gen = DataGenerator(test_seq, test_images, batch_size=BATCH_SIZE, dim=(SEQ+LEFT_PAD, IMG_H, IMG_W), n_channels=IMG_C)
config = {

    "trainable": False,

    "max_len": 70,

    "max_features": 95000,

    "embed_size": 300,

    "spatial_dr": 0.1,

    "dr": 0.1,

    "epochs": 2,

    "num_capsule": 8,

    "dim_capsule": 32,

    "routings": 3,

    "dense_units": 64,

    "units": 64

}
def lstm_capsule(config):

    inp = Input((SEQ+LEFT_PAD, IMG_H, IMG_W, IMG_C))

    x = Conv3D(128, kernel_size=(6, 12, 12), strides=(1,6,6), padding='valid', activation='relu', name='conv1')(inp)

    print(x.shape)

    x = Reshape((SEQ, -1))(x)

    x = LSTM(10, return_sequences=True)(x)

    x = Capsule(num_capsule=config["num_capsule"], dim_capsule=config["dim_capsule"], routings=config["routings"], share_weights=True)(x)

    x = Flatten()(x)

    x = Dropout(config["dr"])(x)

    

    x = Dense(config["dense_units"], activation="relu")(x)

    x = Dense(1)(x)

    

    model = Model(inputs = inp, outputs = x)

    model.compile(

        loss = "mse", 

        optimizer=keras.optimizers.Adam(lr=1e-3))

    

    return model



model = lstm_capsule(config)

model.summary()
mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', save_best_only=True)

hist = model.fit_generator(epochs=15, generator=train_gen, validation_data=test_gen, callbacks=[mc])
'''

input_layer = Input((SEQ+LEFT_PAD, IMG_H, IMG_W, IMG_C))



conv1 = Conv3D(64, kernel_size=(3, 12, 12), strides=(1, 1, 1))(input_layer)

conv1 = MaxPooling3D(pool_size=(1,6,6))(conv1)

conv1 = Activation('relu')(conv1)

conv1 = BatchNormalization()(conv1)

conv1 = Dropout(keep_prob)(conv1)



conv2 = Conv3D(64, kernel_size=(2, 5, 5), strides=(1, 1, 1))(conv1)

conv2 = MaxPooling3D(pool_size=(1,2,2))(conv2)

conv2 = Activation('relu')(conv2)

conv2 = BatchNormalization()(conv2)

conv2 = Dropout(keep_prob)(conv2)



conv3 = Conv3D(64, kernel_size=(2, 5, 5), strides=(1, 1, 1))(conv2)

conv3 = Activation('relu')(conv3)



conv_31 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1))(conv1)

conv_31 = MaxPooling3D(pool_size=(1,3,3))(conv_31)

conv_31 = Activation('relu')(conv_31)

conv_31 = Add()([conv3, conv_31])

conv_31 = BatchNormalization()(conv_31)

conv_31  = Dropout(keep_prob)(conv_31)



conv4 = Conv3D(64, kernel_size=(2, 6, 6), strides=(1, 1, 1))(conv_31)

conv4 = Activation('relu')(conv4)



conv_42 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1))(conv2)

conv_42 = MaxPooling3D(pool_size=(1,3,3))(conv_42)

conv_42 = Activation('relu')(conv_42)

conv_42 = Add()([conv4, conv_42])

conv_42 = BatchNormalization()(conv_42)

conv_42  = Dropout(keep_prob)(conv_42)





reshape = Reshape((SEQ, -1))(conv_42)

lstm = LSTM(10)(reshape)

net = Dense(100, activation='relu')(lstm)

net = Dropout(keep_prob)(net)

net = Dense(50, activation='relu')(net)

net = Dropout(keep_prob)(net)

net = Dense(10, activation='relu')(net)

net = Dropout(keep_prob)(net)

net = Dense(1) (net)



model = Model(inputs=input_layer, outputs=net)



optimizer = keras.optimizers.Adam(lr= 0.0005)

model.compile(optimizer=optimizer, loss='mean_squared_error')



model.summary()

'''


#mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', save_best_only=True)

#hist = model.fit_generator(epochs=30, generator=train_gen, validation_data=test_gen, callbacks=[mc])
'''

from IPython.display import HTML

import base64



def create_download_link(df, title = "Download model file", filename = "model.h5"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = f'<a target="_blank">{title}</a>'

    return HTML(html)



create_download_link(test_angles)

'''
#model.predict()
# plot train and validation loss

'''

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Train loss vs. Validation loss')

plt.ylabel('LOSS')

plt.xlabel('EPOCH')

plt.legend(['train', 'validation'], loc='upper right')

plt.show()



plt.savefig('lc.png')

'''
'''

X = np.empty((1, SEQ+LEFT_PAD, IMG_H, IMG_W, IMG_C))

# Read angle, and sequence image names

seq = test_seq[20][0]

# Load images from image dictionary

images_a = []

for img_name in seq:

    img = test_images[img_name]

    images_a.append(img)

# Convert images to numpy array

images = np.array(images_a).reshape(-1, SEQ+LEFT_PAD, IMG_H, IMG_W, IMG_C)

images = images.astype('float32')

# Normalize images

images /= 255.0

images = -1 + (2 * images)

X[0,] = images





plt.imshow(cv2.cvtColor((images_a[0]), cv2.COLOR_GRAY2RGB))

'''
del train_images
import math

evaluation = model.evaluate_generator(test_gen)

print("RMSE: {}".format(math.sqrt(evaluation)))

print("MSE: {}".format(evaluation))
predicted = model.predict_generator(test_gen)

df = pd.DataFrame(predicted)

print(df.describe)

print(test_angles[:df.shape[0]].describe())


#print(df.describe())
#plt.figure(figsize=(10,5))

#plt.scatter(test_angles[:predicted.shape[0]], predicted, alpha=.75)
'''

presh = predicted

print(len(presh))

gt = test_angles[:presh.shape[0]]



ns = np.arange(gt.shape[0])

#plt.ylim(-0.1, 0.1)

plt.figure(figsize=(10,5))

plt.plot(ns, gt, color='#777777')

plt.plot(ns, presh, color='b')

plt.legend(['Ground Truth', 'Predicted'], loc='upper right')

#plt.hist(presh)

'''
'''

print(len(test_angles//4))

X = np.empty((len(test_angles//4), SEQ+LEFT_PAD, IMG_H, IMG_W, IMG_C))

y = np.empty((len(test_angles//4)))





for i, pack in enumerate(test_seq):

    # Read angle, and sequence image names

    angle = pack[1]

    print(i, angle, end="\r")

    seq = pack[0]

    # Load images from image dictionary

    images = []

    for img_name in seq:

        img = test_images[img_name]

        images.append(img)

    # Convert images to numpy array

    images = np.array(images).reshape(-1, SEQ+LEFT_PAD, IMG_H, IMG_W, IMG_C)

    images = images.astype('float32')

    # Normalize images

    images /= 255.0

    #images = -1 + (2 * images)

    X[i,] = images

    y[i] = angle

'''
'''

# Display activations

layer_outputs = [layer.output for layer in model.layers]

layer_outputs = layer_outputs[1:]

activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(X)

 

def display_activation(activations, col_size, row_size, act_index): 

    activation = activations[act_index]

    activation_index=0

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size, col_size))

    plt.subplots_adjust(0,0,1,1,0,0)

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(activation[0, 0, :, :, activation_index], cmap='gray')

            ax[row][col].tick_params(axis='x', colors=(0,0,0,0))

            ax[row][col].tick_params(axis='y', colors=(0,0,0,0))

            activation_index += 1

            

display_activation(activations, 8, 8,1)

'''
#!ls /kaggle/working