# Preparing convNet model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import normalize
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

# set tensorBoard name to realtime tracking 
NAME = "Cats-vs-dogs-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME)) 
# LOad data

X = pickle.load(open('../input/pickled-cats-vs-dogs/x.pickle', "rb"))
y = pickle.load(open('../input/pickled-cats-vs-dogs/y.pickle', "rb"))

X = X/255.0 # normalize x
y.append(0)
len(y)
# deine parameters and thier vairance
dense_layers = [0]
layers_sizes = [32]
conv_layers = [3]


for dense_layer in dense_layers:
    for layers_size in layers_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-Conv-{}-nodes-{}-dense-{}".format(conv_layer, layers_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME)) 
            print(NAME)
            
            # define model and layers
            model = Sequential()
            
            model.add(Conv2D(layers_size, (3,3), input_shape= X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPool2D(pool_size=(2, 2)))

            for i in range(conv_layer -1):
                model.add(Conv2D(layers_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPool2D(pool_size=(2, 2)))

            model.add(Flatten()) # this is converts our 3D features to 1D vectors
            
            for i in range(dense_layer):
                model.add(Dense(512))
                model.add(Activation('relu'))
                model.add(Dropout(0.2))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
            model.fit(X, y, batch_size=32, epochs =20, validation_split=0.1, callbacks=[tensorboard])
model.save('64x3-CNN.model')
import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]

# def prepare(filepath):
#     IMG_SIZE = 50
#     img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#     return new_array.reshape(-1, IMG_SIZE, IMG_SIZE,1)
# load model
# model = tf.keras.models.load_model('64x3-CNN.model') 
# predict
# prediction = model.predict([prepare('cat1.jpeg')])
# print(CATEGORIES[int(prediction[0][0])])