import warnings

warnings.filterwarnings(category=FutureWarning, action="ignore")



import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

import numpy as np

import pandas as pd

import datetime as dt



from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.utils import plot_model

from tensorflow.keras.optimizers import RMSprop

from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import backend

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.models import Sequential





%matplotlib inline

# from tensorflow.keras.preprocessing.image import ImageDataGenerator



backend.set_image_data_format('channels_last')

DATA_PATH = '../input/'

SERIES = 'A'  # main change indicator

VERSION = 1  # version in series indicator



# LOGDIR = 'S:\\python\\TB_LOG\\DigitFinal\\{:s}{:d}'.format(

#     SERIES, VERSION)  # path to log for TensorBoard



print('{:s}{:d}'.format(SERIES, VERSION))  # version signature

train_data = pd.read_csv(DATA_PATH+'train.csv')

train_data.head()
test_data = pd.read_csv(DATA_PATH+'test.csv')

test_data.index = ([x+1 for x in range(test_data.shape[0])])

print(test_data.shape)

test_data.head()

# suporting methods used to extract and store selected model parameters

def get_model_params(layers) -> str:

    res = {}

    for layer in layers:

        lres = {}

        config = layer.get_config()

        for key in ['filters', 'kernel_size', 'activation', 'pool_size',

                    'padding', 'strides', 'rate', 'units', 'kernel_regularizer',

                    'batch_input_shape']:

            if key in config.keys():

                lres[key] = config[key]

        res[layer.get_config()['name']] = lres

    rep = ''

    for elem in res:

        rep += elem+':  '

        rep += str(res[elem])+'<br>'

    return rep

ncols = 25

nrows = 15

n_images = ncols*nrows

for df, startcol in zip([train_data, test_data], [1, 0]):

    img_idxs = [np.random.randint(0, df.shape[0]) for x in range(n_images)]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,

                           figsize=(ncols*0.7, nrows*0.7))

    axs = ax.reshape(1, -1)[0]



    for img, ax in zip(img_idxs, axs):

        img_data = df.iloc[img, startcol:]

        ax.imshow(X=img_data.values.reshape(28, 28))

        ax.frameon = False

        ax.axis('off')

plt.figure(figsize=(8, 4))

dist = train_data['label'].value_counts(normalize=True, sort=False)

dist.plot(kind='barh');

x_train = train_data.iloc[:, 1:].values.reshape(

    (train_data.shape[0], 28, 28, 1)).astype('float32')

x_train = x_train / 255.0



x_test = test_data.values.reshape(

    (test_data.shape[0], 28, 28, 1)).astype('float32')

x_test = x_test / 255.0



# making response variable categorical

lb = LabelBinarizer()

y_train_ = lb.fit_transform(train_data.iloc[:, 0])

# image generators to feed up a model

train_gen = ImageDataGenerator(

#     samplewise_center=True,

#     samplewise_std_normalization=True,

    rotation_range=9,

    zoom_range=0.09,

    width_shift_range=0.09,

    height_shift_range=0.11,

    validation_split=0.05



)



train_gen.fit(x_train)

train_iterator = train_gen.flow(

    x=x_train, y=y_train_, batch_size=256, subset='training')

val_iterator = train_gen.flow(

    x=x_train, y=y_train_, batch_size=256, subset='validation')

train_x, train_y = train_iterator.next()

# Model

model = Sequential([

    # First convolution

    Conv2D(filters=128, kernel_size=(3, 3),

           activation='relu', input_shape=(28, 28, 1)),

    MaxPooling2D(pool_size=(2, 2)),



    # Second convolution

    Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),



    # Third convolution

    Conv2D(filters=512, kernel_size=(4, 4), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),



    Flatten(),



    # First hidden layer

    Dense(units=256, activation='relu'),

    Dropout(rate=0.35),



    # Second hidden layer

    Dense(units=128, activation='relu'),



    # Third hidden layer

    Dense(units=64, activation='relu'),



    # Output layer

    Dense(10, activation='softmax')

])

print(model.summary())

plot_model(model, show_shapes=True, show_layer_names=True,)

NEPOCHS = 300



# Callbacks

# TensorBoard logging callback

# tensor_board_cb = TensorBoard(log_dir=LOGDIR, write_graph=False,

#                               profile_batch=0)

# iteration cut on plateau

early_stopping_cb = EarlyStopping(monitor='val_acc', min_delta=1e-5,

                                  patience=15, restore_best_weights=True)

# step reduction on p[lateau]

rl_reduce = ReduceLROnPlateau(monitor='val_loss', patience=10,factor=0.25,verbose=1,min_delta=1e-5)



# optimizer

opt_rms = RMSprop(learning_rate=1e-3, centered=False)



model.compile(loss='categorical_crossentropy', optimizer=opt_rms,

              metrics=['accuracy'])



# Training

start = dt.datetime.now()

history = model.fit_generator(

    generator=train_iterator,

    verbose=1,

    epochs=NEPOCHS,

    max_queue_size=10,

    validation_data=val_iterator,

    callbacks=[

#         tensor_board_cb,

        early_stopping_cb,

        rl_reduce

    ]



)



# sending model hyperparameters to Tensorboard (not needed in TF 2.0 Beta)

# summary_ = tf.summary.text('HParams_{:s}{:d}'.format(SERIES, VERSION),

#                            tf.convert_to_tensor('{}'.format(

#                                get_model_params(model.layers))))

# with tf.Session() as sess:

#     summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)

#     text = sess.run(summary_)

#     summary_writer.add_summary(text)

# summary_writer.close()





# Model evaluation

print('\n Evaluation :')

model.evaluate_generator(generator=val_iterator,

                         verbose=1,

#                          callbacks=[tensor_board_cb]

                         )



print('Finished in: ', dt.datetime.now()-start)

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(15, 8))

plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy, {:.4f} '.format(np.max(val_acc)))

plt.legend(loc=0)

plt.ylim(0.7, 1.05)

plt.grid(b=True, which='both')

plt.show()

VERSION += 1

print(dt.datetime.now()-start)

# there is a problem with using ImageDataGenerator to feed the model

# with predict_generator, so here ImageDataGenerator is

# used only as preprocessor



test_gen = ImageDataGenerator(

#     samplewise_center=True,

#     samplewise_std_normalization=True,

    rotation_range=9,

    zoom_range=0.09,

    width_shift_range=0.09,

    height_shift_range=0.11,

)

test_gen.fit(x_test)

test_iterator = test_gen.flow(x=x_test, batch_size=len(x_test), shuffle=False)

test_x = test_iterator.next()

test_x

res = model.predict(test_x)

y_pred = pd.DataFrame([test_data.index, [x.argmax() for x in res]]).T

y_pred.columns = ['ImageId', 'Label']



y_pred.to_csv('digit_submission_2.csv', index=False)
