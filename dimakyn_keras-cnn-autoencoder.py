!unzip ../input/denoising-dirty-documents/sampleSubmission.csv.zip

!unzip ../input/denoising-dirty-documents/test.zip

!unzip ../input/denoising-dirty-documents/train.zip

!unzip ../input/denoising-dirty-documents/train_cleaned.zip
%matplotlib inline



import os

import glob



import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras.models import Model

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.callbacks import EarlyStopping



from keras.layers import Input

from keras.layers import Dense

from keras.layers import Activation

from keras.layers import BatchNormalization

from keras.layers import Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dropout

from keras.layers import UpSampling2D

from keras.layers import Reshape



from sklearn.model_selection import train_test_split
def im_show(im_name):

    plt.figure(figsize=(20,8))

    img = cv2.imread(im_name, 0)

    plt.imshow(img, cmap="gray")

    print(f"[INFO] Image shape: {img.shape} ")

im_show("train/77.png")

im_show("train_cleaned/77.png")
TRAIN_IMAGES = glob.glob('train/*.png')

CLEAN_IMAGES = glob.glob('train_cleaned/*.png')

TEST_IMAGES = glob.glob('test/*.png')

print(f"[INFO] Number of train pictures: {len(TRAIN_IMAGES)}")

print(f"[INFO] Number of train_cleaned pictures: {len(CLEAN_IMAGES)}")

print(f"[INFO] Number of test pictures: {len(TEST_IMAGES)}")
IMG_W = 258

IMG_H = 540

BS = 20

EPOCHS = 200
def load_image(path):

    image_list = np.zeros((len(path), IMG_W, IMG_H, 1))

    for i, fig in enumerate(path):

        img = image.load_img(fig, color_mode='grayscale', target_size=(IMG_W, IMG_H))

        x = image.img_to_array(img).astype('float32')

        x = x / 255.0

        image_list[i] = x

    

    return image_list
x_train = load_image(TRAIN_IMAGES)

y_train = load_image(CLEAN_IMAGES)

x_test = load_image(TEST_IMAGES)



print(f"[INFO] x_train shape: {x_train.shape}")

print(f"[INFO] y_train shape: {y_train.shape}")

print(f"[INFO] x_test shape: {x_test.shape}")
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, 

                                                  random_state=42)

print(f"[INFO] train shape: {x_train.shape} and {y_train.shape}")

print(f"[INFO] validation shape: {x_val.shape} and {y_val.shape}")
def create_deep_conv_ae():

    input_layer = Input(shape=(IMG_W, IMG_H, 1))



    # encoder

    h = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)

    h = MaxPooling2D((2, 2), padding='same')(h)



    # decoder

    h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)

    h = UpSampling2D((2, 2))(h)

    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h)



    return Model(input_layer, output_layer)



autoencoder = create_deep_conv_ae()

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()
early_stopping = EarlyStopping(monitor='val_loss',

                                       min_delta=0,

                                       patience=5,

                                       verbose=1, 

                                       mode='auto')
history = autoencoder.fit(x_train, y_train,

                         batch_size=BS,

                         epochs=EPOCHS,

                         validation_data=(x_val, y_val),

                         callbacks=[early_stopping]

                        )
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
preds = autoencoder.predict(x_test)
def plot_digits(*args):

    args = [x.squeeze() for x in args]

    n = min([x.shape[0] for x in args])

    

    plt.figure(figsize=(2*n, 2*len(args)))

    for j in range(n):

        for i in range(len(args)):

            ax = plt.subplot(len(args), n, i*n + j + 1)

            plt.imshow(args[i][j])

            plt.gray()

            ax.get_xaxis().set_visible(False)

            ax.get_yaxis().set_visible(False)



    plt.show()



plot_digits(x_test[:5], preds[:5])
ids = []

vals = []

for i, f in enumerate(TEST_IMAGES):

    file = os.path.basename(f)

    imgid = int(file[:-4])

    test_img = cv2.imread(f, 0)

    img_shape = test_img.shape

    print('processing: {}'.format(imgid))

    print(img_shape)

    preds_reshaped = cv2.resize(preds[i], (img_shape[1], img_shape[0]))

    for r in range(img_shape[0]):

        for c in range(img_shape[1]):

            ids.append(str(imgid)+'_'+str(r + 1)+'_'+str(c + 1))

            vals.append(preds_reshaped[r, c])



print('Writing to csv file')

pd.DataFrame({'id': ids, 'value': vals}).to_csv('submission.csv', index=False)