import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm_notebook as tqdm

import scipy

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score

from IPython.display import clear_output





import keras

from keras.applications import VGG16, DenseNet121

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

from keras import layers





print(os.listdir('../input/'))

%matplotlib inline

train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print(train_df.shape)

print(test_df.shape)

train_df.head()
def preprocess_image(image_path, desired_size=300):

    im = cv2.imread(image_path)

    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    im = cv2.resize(im, (desired_size, desired_size))

    im = cv2.addWeighted(im, 4, cv2.blur(im, ksize=(10,10)), -4, 128)

    return im
N = train_df.shape[0]

x_train = np.empty((N, 300, 300, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(train_df['id_code'])):

    x_train[i, :, :, :] = preprocess_image(

        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'

    )
N = test_df.shape[0]

x_test = np.empty((N, 300, 300, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(test_df['id_code'])):

    x_test[i, :, :, :] = preprocess_image(

        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'

    )
y_train = pd.get_dummies(train_df['diagnosis']).values



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)



plt.imshow(x_train[0])
y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)

y_train_multi[:, 4] = y_train[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])



print("Original y_train:", y_train.sum(axis=0))

print("Multilabel version:", y_train_multi.sum(axis=0))
x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train, 

    test_size=0.15, 

    random_state=2019

)
BATCH_SIZE = 32



def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.10,  # set range for random zoom

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True,  # randomly flip images

    )



# Using original generator

data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE)
class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_kappas = []



    def on_epoch_end(self, epoch, logs={}):

        X_val, y_val = self.validation_data[:2]

        y_pred = self.model.predict(X_val)



        _val_kappa = cohen_kappa_score(

            y_val.argmax(axis=1), 

            y_pred.argmax(axis=1), 

            weights='quadratic'

        )



        self.val_kappas.append(_val_kappa)



        print(f"val_kappa: {_val_kappa:.4f}")



        return
vgg = VGG16(

    weights='../input/vgg16imagenetnotop/vgg16-notop.h5',

    include_top=False,

    input_shape=(300, 300, 3)

)
def build_model():

    model = Sequential()

    model.add(vgg)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='softmax'))

    

    model.compile(

        loss='categorical_crossentropy',

        optimizer=Adam(lr=0.00005),

        metrics=['accuracy']

    )

    return model
model = build_model()

model.summary()
class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.i += 1

        

        clear_output(wait=True)

        plt.plot(self.x, self.losses, label="loss")

        plt.plot(self.x, self.val_losses, label="val_loss")

        plt.legend()

        plt.show();

        

plot_losses = PlotLosses()
kappa_metrics = Metrics()



checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_loss', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



history = model.fit_generator(

    data_generator,

    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,

    epochs=50,

    validation_data=(x_val, y_val),

    callbacks=[checkpoint, kappa_metrics, plot_losses]

)
model.load_weights('model.h5')

y_test = model.predict(x_test, verbose=2)



test_df['diagnosis'] = y_test.argmax(axis=1)

print(test_df.shape)



test_df.to_csv('submission.csv',index=False)
