import os

import glob



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import cv2



from keras.preprocessing import image

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D

from keras.layers import MaxPooling2D, Dropout, UpSampling2D
%matplotlib inline
plt.figure(figsize=(20,8))

img = cv2.imread('../input/train/101.png', 0)

plt.imshow(img, cmap='gray')

print(img.shape)
TRAIN_IMAGES = glob.glob('../input/train/*.png')

CLEAN_IMAGES = glob.glob('../input/train_cleaned/*.png')

TEST_IMAGES = glob.glob('../input/test/*.png')
def load_image(path):

    image_list = np.zeros((len(path), 258, 540, 1))

    for i, fig in enumerate(path):

        img = image.load_img(fig, color_mode='grayscale', target_size=(258, 540))

        x = image.img_to_array(img).astype('float32')

        x = x / 255.0

        image_list[i] = x

    

    return image_list
x_train = load_image(TRAIN_IMAGES)

y_train = load_image(CLEAN_IMAGES)

x_test = load_image(TEST_IMAGES)



print(x_train.shape, x_test.shape)
def train_val_split(x_train, y_train):

    rnd = np.random.RandomState(seed=42)

    perm = rnd.permutation(len(x_train))

    train_idx = perm[:int(0.8 * len(x_train))]

    val_idx = perm[int(0.8 * len(x_train)):]

    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]



x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)

print(x_train.shape, x_val.shape)
class Autoencoder():

    def __init__(self):

        self.img_rows = 258

        self.img_cols = 540

        self.channels = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        

        optimizer = Adam(lr=0.001)

        

        self.autoencoder_model = self.build_model()

        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)

        self.autoencoder_model.summary()

    

    def build_model(self):

        input_layer = Input(shape=self.img_shape)

        

        # encoder

        h = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)

        h = MaxPooling2D((2, 2), padding='same')(h)

        

        # decoder

        h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)

        h = UpSampling2D((2, 2))(h)

        output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h)

        

        return Model(input_layer, output_layer)

    

    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=20):

        early_stopping = EarlyStopping(monitor='val_loss',

                                       min_delta=0,

                                       patience=5,

                                       verbose=1, 

                                       mode='auto')

        history = self.autoencoder_model.fit(x_train, y_train,

                                             batch_size=batch_size,

                                             epochs=epochs,

                                             validation_data=(x_val, y_val),

                                             callbacks=[early_stopping])

        plt.plot(history.history['loss'])

        plt.plot(history.history['val_loss'])

        plt.title('Model loss')

        plt.ylabel('Loss')

        plt.xlabel('Epoch')

        plt.legend(['Train', 'Test'], loc='upper left')

        plt.show()

    

    def eval_model(self, x_test):

        preds = self.autoencoder_model.predict(x_test)

        return preds
ae = Autoencoder()

ae.train_model(x_train, y_train, x_val, y_val, epochs=200, batch_size=20)
preds = ae.eval_model(x_test)
preds_0 = preds[10] * 255.0

preds_0 = preds_0.reshape(258, 540)

x_test_0 = x_test[10] * 255.0

x_test_0 = x_test_0.reshape(258, 540)

plt.imshow(x_test_0, cmap='gray')
plt.imshow(preds_0, cmap='gray')
preds = preds.reshape(-1, 258, 540)
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
print(len(ids))