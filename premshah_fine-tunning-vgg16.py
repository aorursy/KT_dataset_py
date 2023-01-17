# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainLabels = pd.read_csv("../input/trainLabels.csv")

trainLabels.head()
import os



listing = os.listdir("../input") 

listing.remove("trainLabels.csv")

np.size(listing)
from PIL import Image

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image



# input image dimensions

img_rows, img_cols = 224, 224



immatrix = []

imlabel = []



for file in listing:

    base = os.path.basename("../input/" + file)

    fileName = os.path.splitext(base)[0]

    imlabel.append(trainLabels.loc[trainLabels.image==fileName, 'level'].values[0])

    im = Image.open("../input/" + file)

    img = im.resize((img_rows,img_cols))

    #img4d = np.expand_dims(img, axis=0)

    #img4d = preprocess_input(img4d)

    immatrix.append(np.array(img))
immatrix = np.asarray(immatrix)

imlabel = np.asarray(imlabel)
from sklearn.utils import shuffle



data,Label = shuffle(immatrix,imlabel, random_state=2)

train_data = [data,Label]

type(train_data)
(X, y) = (train_data[0],train_data[1])
from sklearn.cross_validation import train_test_split



# STEP 1: split X and y into training and testing sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)



print(X_train.shape)

print(X_test.shape)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



X_train /= 255

X_test /= 255



print('X_train shape:', X_train.shape)

print(X_train.shape[0], 'train samples')

print(X_test.shape[0], 'test samples')
from keras.utils import np_utils



# number of output classes

nb_classes = 5



# convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(y_train, nb_classes)

Y_test = np_utils.to_categorical(y_test, nb_classes)



i = 100

plt.imshow(X_train[i, 0], interpolation='nearest')

print("label : ", Y_train[i,:])
from sklearn.cross_validation import train_test_split



# STEP 1: split X and y into training and testing sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)



print(X_train.shape)

print(X_test.shape)
from keras.utils import np_utils



# convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(y_train, nb_classes)

Y_test = np_utils.to_categorical(y_test, nb_classes)
from keras.applications.vgg16 import VGG16



vgg16_model = VGG16(weights="imagenet", include_top=True)

 

    #visualize layers

print("VGG16 model layers")

for i, layer in enumerate(vgg16_model.layers):

    print(i, layer.name, layer.output_shape)
from keras.models import Model, load_model



# (2) remove the top layer

base_model = Model(input=vgg16_model.input, 

                   output=vgg16_model.get_layer("block5_pool").output)

from keras.layers import Dense, Dropout, Reshape



# (3) attach a new top layer

base_out = base_model.output

base_out = Reshape((25088,))(base_out)

top_fc1 = Dense(256, activation="relu")(base_out)

top_fc1 = Dropout(0.5)(top_fc1)

# output layer: (None, 5)

top_preds = Dense(5, activation="softmax")(top_fc1)
# (4) freeze weights until the last but one convolution layer (block4_pool)

for layer in base_model.layers[0:14]:

    layer.trainable = False
# (5) create new hybrid model

model = Model(input=base_model.input, output=top_preds)
from keras.optimizers import SGD



BATCH_SIZE = 32

NUM_EPOCHS = 5



# (6) compile and train the model

sgd = SGD(lr=1e-4, momentum=0.9)

model.compile(optimizer=sgd, loss="categorical_crossentropy",

              metrics=["accuracy"])



history = model.fit([X_train], [Y_train], nb_epoch=NUM_EPOCHS, 

                    batch_size=BATCH_SIZE, validation_split=0.1, 

                    callbacks=[checkpoint])
# evaluate final model

Ytest = model.predict(X_test)