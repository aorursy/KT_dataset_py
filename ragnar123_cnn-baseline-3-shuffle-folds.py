import numpy as np 

import pandas as pd

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

sub = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

print('Our train set have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

print('Our test set have {} rows and {} columns'.format(test.shape[0], test.shape[1]))
# target variable distribution

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize = (10,8))

sns.countplot(x = 'label', data = train)

plt.show()
train.isnull().sum().sum()
from sklearn.model_selection import train_test_split

def preprocessing(train, test):

    # drop label column of the train set and reshape, in this case we have 28X28 pixel images

    IMG_SIZE = 28

    img_train = train.drop(['label'], axis = 1).values.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')

    img_test = test.drop(['id'], axis = 1).values.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')

    img_y = train['label'].values

    # scale data (rgb goes from 0 to 255, dividing by 255 change the range to 0-1)

    img_train /= 255

    img_test /= 255

    # taking 20% of our train data as eval data.

    x_train, x_val, y_train, y_val = train_test_split(img_train, img_y, test_size = 0.20)

    print('Our transformed train set have the following dimension: ', x_train.shape)

    print('Our transformed valid set have the following dimension: ', x_val.shape)

    return img_test, x_train, x_val, y_train, y_val
import keras

from keras.layers import Input, Dense, Dropout, Flatten

from keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization

from keras import backend as K

from keras.callbacks import ModelCheckpoint

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

# build model

n_classes = train['label'].value_counts().count()

# model from https://www.kaggle.com/anshumandec94/6-layer-conv-nn-using-adam

def build_model(input_shape=(28, 28, 1), classes = n_classes):

    input_layer = Input(shape=input_shape)

    x = Conv2D(16, (3,3), strides=1, padding="same", name="conv1")(input_layer)

    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform", name="batch1")(x)

    x = Activation('relu',name='relu1')(x)

    x = Dropout(0.1)(x)

    

    x = Conv2D(32, (3,3), strides=1, padding="same", name="conv2")(x)

    x = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch2")(x)

    x = Activation('relu',name='relu2')(x)

    x = Dropout(0.15)(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name="max2")(x)

    

    x = Conv2D(64, (5,5), strides=1, padding ="same", name="conv3")(x)

    x = BatchNormalization(momentum=0.17, epsilon=1e-5, gamma_initializer="uniform", name="batch3")(x)

    x = Activation('relu', name="relu3")(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name="max3")(x)

    

    x = Conv2D(128, (5,5), strides=1, padding="same", name="conv4")(x)

    x = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch4")(x)

    x = Activation('relu', name="relu4")(x)

    x = Dropout(0.17)(x)

    

    x = Conv2D(64, (3,3), strides=1, padding="same", name="conv5")(x)

    x = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch5")(x)

    x = Activation('relu', name='relu5')(x)

    x = Dropout(0.2)(x)

    

    x = Conv2D(32, (3,3), strides=1, padding="same", name="conv6")(x)

    x = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch6" )(x)

    

    x = Activation('relu', name="relu6")(x)

    x = Dropout(0.05)(x)

    

    x = Flatten()(x)

    x = Dense(50, name="Dense1")(x)

    x = Activation('relu', name='relu7')(x)

    x = Dropout(0.05)(x)

    x = Dense(25, name="Dense2")(x)

    x = Activation('relu', name='relu8')(x)

    x = Dropout(0.03)(x)

    x = Dense(classes, name="Dense3")(x)

    x = Activation('softmax')(x)



    model = Model(inputs=input_layer, outputs=x)

    return model
# let's create a data generator to make some data augmentation

# let's create a checkpoint callback to save best model in the training process

# let's create a another callback to reduce the learning rate if the validation score dont improve in x round

def trng_lr_ck_opt(modelname):

    train_generator = ImageDataGenerator(rotation_range = 8,  # we dont want to rotate that much (confuse)

                                        zoom_range = 0.28,

                                        width_shift_range = 0.25,

                                        height_shift_range = 0.25)

    learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 5, verbose = 1, factor = 0.5, min_le = 0.000001)

    

    checkpoint = ModelCheckpoint(modelname+'.hdf5', monitor = 'val_accuracy', verbose = 1, save_best_only = True)

    return train_generator, learning_rate_reduction, checkpoint



def compile_model():

    optimizer = Adam(lr = 0.001)

    model = build_model(input_shape = (28, 28, 1), classes = n_classes)

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    return model



# final step, join all the modules and train the model

def train_and_evaluate(train, test, batch_size, epochs):

    img_test, x_train, x_val, y_train, y_val = preprocessing(train, test)

    model = compile_model()

    train_generator, learning_rate_reduction, checkpoint = trng_lr_ck_opt('bestmodel')

    history = model.fit_generator(train_generator.flow(x_train, y_train, batch_size = batch_size),

                                  steps_per_epoch = x_train.shape[0] // batch_size,

                                  epochs = epochs, 

                                  validation_data = (x_val, y_val),

                                  callbacks = [checkpoint, learning_rate_reduction])

    return img_test, history



# run train and evaluate

BATCH_SIZE = 64

EPOCHS = 70

img_test, history = train_and_evaluate(train, test, BATCH_SIZE, EPOCHS)
def plot_loss_acc(his, epoch):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15,10))

    ax1.plot(np.arange(0, epoch), his.history['loss'], label = 'train_loss')

    ax1.plot(np.arange(0, epoch), his.history['val_loss'], label = 'val_loss')

    ax1.set_title('Loss')

    ax1.figure.legend()

    ax2.plot(np.arange(0, epoch), his.history['accuracy'], label = 'train_acc')

    ax2.plot(np.arange(0, epoch), his.history['val_accuracy'], label = 'val_accuracy')

    ax2.set_title('Accuracy')

    ax2.figure.legend()

    plt.show()

plot_loss_acc(history, EPOCHS)
# using a validation of 15%, 3 shuffle partitions

from sklearn import metrics

def train_and_evaluate_kfold(train, test, batch_size, epochs):

    IMG_SIZE = 28

    img_test = test.drop(['id'], axis = 1).values.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255

    img_y = train['label'].values

    train = train.drop(['label'], axis = 1)

    preds = np.zeros([test.shape[0], n_classes])

    for fold_n in [1, 2, 3]:

        print('Training fold {}'.format(fold_n))

        x_train, x_val, y_train, y_val = train_test_split(train, img_y, test_size = 0.15, random_state = 42 + fold_n)

        x_train = x_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255

        x_val = x_val.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255

        model = compile_model()

        train_generator, learning_rate_reduction, checkpoint = trng_lr_ck_opt('model_fold_{}'.format(fold_n))

        history = model.fit_generator(train_generator.flow(x_train, y_train, batch_size = batch_size),

                                      steps_per_epoch = x_train.shape[0] // batch_size,

                                      epochs = epochs, 

                                      validation_data = (x_val, y_val),

                                      callbacks = [checkpoint, learning_rate_reduction])

        model.load_weights('model_fold_{}'.format(fold_n) + '.hdf5')

        preds += model.predict(img_test) / 3

        

    preds = np.argmax(preds, axis = 1)

    return preds



# let's train the models, load the best weight for each fold and predict the test set

preds = train_and_evaluate_kfold(train, test, BATCH_SIZE, EPOCHS)
# save predictions

sub['label'] = preds

sub.to_csv('cnn_kfold_baseline.csv', index = False)