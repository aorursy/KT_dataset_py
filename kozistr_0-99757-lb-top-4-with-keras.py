import pandas as pd

import numpy as np

import tensorflow as tf



from sklearn.model_selection import train_test_split



from keras.utils.np_utils import to_categorical

from keras.models import Model

from keras.regularizers import l2

from keras.layers.core import Dense, Activation

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D

from keras.layers import Flatten, Conv2D, ZeroPadding2D, Dropout, Input

from keras.optimizers import Adam, SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# reproducibility

seed = 777

tf.set_random_seed(seed)

np.random.seed(seed)



epochs = 150  # 50, 100, 150

batch_size = 64  # 32, 48, 64, 84

isFT = False  # true # for fine-tuning ?
# prepare train & test data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



y_train = train['label']                           # label

y_train = to_categorical(y_train, num_classes=10)  # ten classes : 0 ~ 9 # label encoding (one-hot)

x_train = train.drop(labels=['label'], axis=1)     # remove label lefts are image-datas



del train
# normalize # the CNN coverge faster on (0-1) then (0-255) data

x_train /= 255.

test /= 255.
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)



# reshape # (-1, 784) -> (-1, 28, 28, 1)

x_train = x_train.values.reshape(-1, 28, 28, 1)

x_valid = x_valid.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
def conv2d_bn(nb_filter, weight_decay=1e-4, name=''):

    def f(x):

        x = Conv2D(nb_filter, (3, 3), activation=None,

                   kernel_initializer="he_uniform",

                   kernel_regularizer=l2(weight_decay),

                   padding="same", name=name)(x)

        x = BatchNormalization(gamma_regularizer=l2(weight_decay),

                               beta_regularizer=l2(weight_decay))(x)

        return Activation('relu')(x)  # ZeroPadding2D((1, 1))(x)



    return f





def vgg_like(fine_tune=False):

    def conv_pool(x, F, name, pad='valid'):

        for idx, f in enumerate(F):

            x = conv2d_bn(f, name='{}_conv{}'.format(name, str(idx)))(x)



        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=pad, name='{}_pool'.format(name))(x)

        return Dropout(0.25)(x)



    net_in = Input(shape=(28, 28, 1))



    net = conv_pool(net_in, [64, 64], "block1")

    net = conv_pool(net, [128, 128], "block2")

    net = conv_pool(net, [256, 256, 256], "block3")

    net = conv_pool(net, [512, 512, 512], "block4")

    # net = conv_pool(net, [512, 512, 512], "block5", pad='same')



    net = Flatten()(net)

    net = Dense(512, activation='relu', name='fc-1')(net)

    # net = Dropout(0.5)(net)

    net = Dense(10, activation='softmax', name='predictions')(net)



    net = Model(inputs=net_in, outputs=net)



    if fine_tune:

        for layer in net.layers[:16]:

            layer.trainable = False



        net.compile(optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),

                    loss="categorical_crossentropy", metrics=["accuracy"])

    else:

        net.compile(optimizer=Adam(lr=9.95e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-9),

                    loss="categorical_crossentropy", metrics=["accuracy"])



    return net
preds_test = []

models = [vgg_like, vgg_like]  # ensemble models



for i, model in enumerate(models):

    model = model(fine_tune=isFT)



    w_fn = 'mnist-1-{}.h5'.format(i)



    if isFT:

        model.load_weights(w_fn)

        w_fn = 'mnist-2-{}.h5'.format(i)



    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=2.5e-5)

    model_checkpoint = ModelCheckpoint(w_fn, monitor='val_acc', save_best_only=True)



    data_generate = ImageDataGenerator(

        rotation_range=20 - 5 * i,

        shear_range=0.1 * i,

        zoom_range=0.1,

        width_shift_range=0.2,

        height_shift_range=0.2,

    )

    data_generate.fit(x_train)



    model.fit_generator(data_generate.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,

                        validation_data=(x_valid, y_valid), verbose=2,

                        steps_per_epoch=x_train.shape[0] // batch_size,

                        callbacks=[learning_rate_reduction, model_checkpoint, early_stopping]

                        )



    model.load_weights(w_fn)



    results = model.predict(test)

    results = np.argmax(results, axis=1)



    preds_test.append(results)
# voting for the cases which use more ensemble models

preds = np.array([])

for i in range(28000):

    tmp = []

    for j in range(len(models)):

        tmp.append(preds_test[j][i])

    n = np.bincount(tmp).argmax()

    preds = np.append(preds, n)



submission = pd.read_csv('./sample_submission.csv')

submission['Label'] = preds

submission['Label'] = submission['Label'].astype(int)



submit_file = 'submit-1.csv'

if isFT:

    submit_file = 'submit-2.csv'

submission.to_csv(submit_file, index=False)