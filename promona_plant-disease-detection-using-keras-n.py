import numpy as np

import pickle

import cv2

from os import listdir

from sklearn.preprocessing import LabelBinarizer

from keras.layers import Input, Convolution2D, GlobalAveragePooling2D, Dense , DepthwiseConv2D

from keras.models import Model

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
EPOCHS = 10

INIT_LR = 1e-3

BS = 32

default_image_size = tuple((256, 256))

image_size = 0

directory_root = '../input/rice-diseases-image-dataset/labelledrice/'

width=256

height=256

depth=3
def convert_image_to_array(image_dir):

    try:

        image = cv2.imread(image_dir)

        if image is not None :

            image = cv2.resize(image, default_image_size)   

            return img_to_array(image)

        else :

            return np.array([])

    except Exception as e:

        print(f"Error : {e}")

        return None
image_list, label_list = [], []

try:

    print("[INFO] Loading images ...")

    root_dir = listdir(directory_root)

    for directory in root_dir :

        # remove .DS_Store from list

        if directory == ".DS_Store" :

            root_dir.remove(directory)



    for plant_folder in root_dir :

        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

        

        for disease_folder in plant_disease_folder_list :

            # remove .DS_Store from list

            if disease_folder == ".DS_Store" :

                plant_disease_folder_list.remove(disease_folder)



        for plant_disease_folder in plant_disease_folder_list:

            print(f"[INFO] Processing {plant_disease_folder} ...")

            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

                

            for single_plant_disease_image in plant_disease_image_list :

                if single_plant_disease_image == ".DS_Store" :

                    plant_disease_image_list.remove(single_plant_disease_image)



            for image in plant_disease_image_list[:]:

                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                    image_list.append(convert_image_to_array(image_directory))

                    label_list.append(plant_disease_folder)

    print("[INFO] Image loading completed")  

except Exception as e:

    print(f"Error : {e}")
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

label_list = labelencoder.fit_transform(label_list)

import numpy as np

import keras



class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, image_lists, label_lists, batch_size=32, dim=(256, 256), n_channels=3,

                 n_classes=4, shuffle=True):

        'Initialization'

        self.dim = dim

        self.batch_size = batch_size

#         self.labels = labels

        self.image_lists = image_lists

        self.label_lists = label_lists

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.image_lists) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        



        # Find list of IDs

        image_list_IDs_temp = [self.image_lists[k] for k in indexes]



        # Generate data

        X, y = self.__data_generation(image_list_IDs_temp)

        return X,y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.image_lists))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size,), dtype=int)

        



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store sample

           

            X[i,] = ID

            # Store class

            y[i] = self.label_lists[i]



        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

print("[INFO] Spliting data to train, test")

x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state = 42) 
# Creating dataloader object

train_loader = DataGenerator(x_train,y_train)

validation_loader = DataGenerator(x_test,y_test)
aug = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
model = Sequential()

inputShape = (height, width, depth)

chanDim = -1

if K.image_data_format() == "channels_first":

    inputShape = (depth, height, width)

    chanDim = 1

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation("relu"))

model.add(BatchNormalization())

# model.add(Dropout(0.5))

model.add(Dense(4))

model.add(Activation("softmax"))
#Mobilenet from scratch 

activation = 'relu'

inputs = Input((256, 256, 3))

x = Convolution2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation=activation)(

            inputs)

x = BatchNormalization()(x)

x = Dropout(0.25)(x)



x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



for _ in range(5):

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

    x = BatchNormalization()(x)



    x = Convolution2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(

                x)

    x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = Convolution2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)

x = Convolution2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation=activation)(x)

x = BatchNormalization()(x)



x = GlobalAveragePooling2D()(x)

out = Dense(4, activation='softmax')(x)



model = Model(inputs, out)
# VGG 16 

model = Sequential([

Conv2D(64, (3, 3), input_shape=(256,256,3), padding='same', activation='relu'),

Conv2D(64, (3, 3), padding='same', activation='relu'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(128, (3, 3), padding='same', activation='relu'),

Conv2D(128, (3, 3), padding='same', activation='relu'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(256, (3, 3), padding='same', activation='relu'),

Conv2D(256, (3, 3), padding='same', activation='relu'),

Conv2D(256, (3, 3), padding='same', activation='relu'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(512, (3, 3), padding='same', activation='relu'),

Conv2D(512, (3, 3), padding='same', activation='relu'),

Conv2D(512, (3, 3), padding='same', activation='relu'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(512, (3, 3), padding='same', activation='relu'),

Conv2D(512, (3, 3), padding='same', activation='relu'),

Conv2D(512, (3, 3), padding='same', activation='relu'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Flatten(),

Dense(4096, activation='relu'),

Dropout(0.5),

Dense(4096, activation='relu'),

Dropout(0.5),

Dense(4, activation='softmax')

])
model.summary()
from keras.optimizers import SGD

opt = SGD(lr=0.001, decay=INIT_LR / EPOCHS)

# lr=INIT_LR, decay=INIT_LR / EPOCHS

# distribution

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network

print("[INFO] training network...")
# history = model.fit_generator(

#     aug.flow(x_train, y_train, batch_size=BS),

#     validation_data=(x_test, y_test),

#     steps_per_epoch=len(x_train) // BS,

#     epochs=EPOCHS, verbose=1

#     )



history = model.fit_generator(

    train_loader, validation_data = validation_loader,

    steps_per_epoch=10,epochs=20,verbose=1)

    
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy

plt.plot(epochs, acc, 'b', label='Training accurarcy')

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()

#Train and validation loss

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()
print("[INFO] Calculating model accuracy")

scores = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {scores[1]*100}")
# save the model to disk

print("[INFO] Saving model...")

pickle.dump(model,open('cnn_model.pkl', 'wb'))