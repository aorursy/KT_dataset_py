import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import matplotlib.image as mplimg

from matplotlib.pyplot import imshow



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from skimage import color

from skimage import io

from skimage.transform import rescale, resize



from keras import layers

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.imagenet_utils import preprocess_input

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D

from keras.models import Model

from keras.applications import Xception

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping

import keras.backend as K

from keras.models import Sequential

import tensorflow as tf

import warnings



#warnings.simplefilter("ignore", category=DeprecationWarning)



#config = tf.ConfigProto()

#config.gpu_options.allow_growth = True

#sess = tf.Session(config=config)

#K.set_session(sess)
img_size = 90

train_df = pd.read_csv(r"../input/train.csv")

train_df.head()
def prepareImages(data, m, dataset):

    print("Preparing images")

    X_train = np.zeros((m, img_size, img_size, 1))

    count = 0

    

    for fig in data['Image']:

        img = image.load_img(r"../input/"+dataset+"/"+fig, target_size=(img_size, img_size, 3))

        x = image.img_to_array(img)

        #x = io.imread(r"../input/"+dataset+"/"+fig)

        x = color.rgb2gray(x)

        #x = resize(x, (img_size, img_size), anti_aliasing=True)

        x = preprocess_input(x)

        x = np.expand_dims(x, axis=2)



        X_train[count] = x

        #if (count%500 == 0):

            #print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return X_train



def prepareLabels(y):

    values = np.array(y)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)

    # print(integer_encoded)



    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # print(onehot_encoded)



    y = onehot_encoded

    # print(y.shape)

    return y, label_encoder
X = prepareImages(train_df, train_df.shape[0], "train")

X /= 255

y,label_encoder=prepareLabels(train_df['Id'])
split = int(0.8*len(X))

X_val = X[split-len(X):]

y_val = y[split-len(X):]

X = X[:split]

y = y[:split]
INIT_LR = 0.0007

EPOCHS = 50

BS = 64

num_classes = y.shape[1]



model = Sequential()



model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (img_size, img_size, 1)))



model.add(BatchNormalization(axis = 3, name = 'bn0'))

model.add(Activation('relu'))



model.add(MaxPooling2D((2, 2), name='max_pool'))

model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))

model.add(Activation('relu'))

model.add(AveragePooling2D((3, 3), name='avg_pool'))



model.add(GlobalAveragePooling2D())

model.add(Dense(500, activation="relu", name='rl'))

model.add(Dropout(0.8))

model.add(Dense(y.shape[1], activation='softmax', name='sm'))



#model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])







aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.3, height_shift_range=0.3, shear_range=0.1, zoom_range=0.3, horizontal_flip=True, vertical_flip=False, fill_mode="nearest")

#model = Model(input = base_model.input, output = predictions)

#model.summary()

#file_path=r"../happy_whale.hdf5"

#model.load_weights(file_path)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=2, mode='auto', baseline=0, restore_best_weights=True)

#checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

callbacks_list = [early_stopping]
history=model.fit_generator(aug.flow(X, y, BS),epochs=EPOCHS,validation_data=aug.flow(X_val, y_val, BS),verbose=2)
test = os.listdir(r"../input/test/")

col = ['Image']

test_df = pd.DataFrame(test, columns=col)

test_df['Id'] = ''

Z = prepareImages(test_df, test_df.shape[0], "test")

Z /= 255
predictions = model.predict(np.array(Z), verbose=1)

for i, pred in enumerate(predictions):

    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_df.head(10)

test_df.to_csv('submission.csv', index=False)