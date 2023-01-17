import numpy as np
import os
import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import keras
from keras.preprocessing.image import load_img
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.layers import Reshape, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.regularizers import l1_l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

%matplotlib inline
%env JOBLIB_TEMP_FOLDER=/tmp
dataset = 'notMNIST_small'
DATA_PATH = '../input/' + dataset + '/' + dataset
max_images = 100
grid_width = 10
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
classes = os.listdir(DATA_PATH)
for j, cls in enumerate(classes):
    figs = os.listdir(DATA_PATH + '/' + cls)
    for i, fig in enumerate(figs[:grid_width]):
        ax = axs[j, i]
        ax.imshow(np.array(load_img(DATA_PATH + '/' + cls + '/' + fig)))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
X = []
labels = []
# for each folder (holding a different set of letters)
for directory in os.listdir(DATA_PATH):
    # for each image
    for image in os.listdir(DATA_PATH + '/' + directory):
        # open image and load array data
        try:
            file_path = DATA_PATH + '/' + directory + '/' + image
            img = Image.open(file_path)
            img.load()
            img_data = np.asarray(img, dtype=np.int16)
            # add image to dataset
            X.append(img_data)
            # add label to labels
            labels.append(directory)
        except:
            None # do nothing if couldn't load file
N = len(X) # number of images
img_size = len(X[0]) # width of image
X = np.asarray(X).reshape(N, img_size, img_size,1) # add our single channel for processing purposes
labels_cat = to_categorical(list(map(lambda x: ord(x)-ord('A'), labels)), 10) # convert to one-hot
labels = np.asarray(list(map(lambda x: ord(x)-ord('A'), labels)))
cls_s = np.sum(labels,axis=0)

fig, ax = plt.subplots()
plt.bar(np.arange(10), cls_s)
plt.ylabel('No of pics')
plt.xticks(np.arange(10), np.sort(classes))
plt.title('Checking balance for data set..')
plt.show()
from sklearn.cross_validation import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(X,labels,test_size=0.2)
X_train_cat,X_valid_cat,y_train_cat,y_valid_cat=train_test_split(X,labels_cat,test_size=0.2)

print('Training:', X_train.shape, y_train.shape)
print('Validation:', X_valid.shape, y_valid.shape)
# def randomize(dataset, labels):
#   permutation = np.random.permutation(labels.shape[0])
#   shuffled_dataset = dataset[permutation,:,:]
#   shuffled_labels = labels[permutation]
#   return shuffled_dataset, shuffled_labels
#   
# X_train, y_train = randomize(X_train, y_train)
# X_test, y_test = randomize(X_test, y_test)
# X_train_cat, y_train_cat = randomize(X_train_cat, y_train_cat)
# X_test_cat, y_test_cat = randomize(X_test_cat, y_test_cat)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
for j in range(max_images):
    ax = axs[int(j/grid_width), j%grid_width]
    ax.imshow(X_train[j,:,:,0])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
samples, width, height = np.squeeze(X_train).shape
X_train_lr = np.reshape(X_train,(samples,width*height))
y_train_lr = y_train

samples, width, height = np.squeeze(X_valid).shape
X_valid_lr = np.reshape(X_valid,(samples,width*height))
y_valid_lr = y_valid
lr = LogisticRegression(multi_class='ovr', solver='saga', random_state=42, 
                        verbose=0, max_iter=5000, n_jobs=-1)
lr.fit(X_train_lr, y_train_lr)
lr.score(X_train_lr, y_train_lr)
y_pred_lr = lr.predict(X_valid_lr)
from sklearn import metrics
metrics.accuracy_score(y_valid_lr, y_pred_lr)
rf = RandomForestClassifier(n_estimators=1000, criterion="gini", max_depth=10, 
                            min_samples_split=2, min_samples_leaf=1, 
                            n_jobs=-1, random_state=42, verbose=0)
rf.fit(X_train_lr, y_train_lr)
rf.score(X_train_lr, y_train_lr)
y_pred_rf = rf.predict(X_valid_lr)
metrics.accuracy_score(y_valid_lr, y_pred_rf)
xg_train = xgb.DMatrix(X_train_lr, label=y_train_lr)
xg_valid = xgb.DMatrix(X_valid_lr, label=y_valid_lr)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax' # 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 10
param['silent'] = 0
param['nthread'] = 4
param['num_class'] = 10

watchlist = [(xg_train, 'train'), (xg_valid, 'test')]
num_round = 20
xg = xgb.train(param, xg_train, num_round, 
               watchlist, early_stopping_rounds=50, 
               maximize=True)

y_pred_xg = xg.predict(xg_valid)
metrics.accuracy_score(y_valid_lr, y_pred_xg)
def build_logistic_model(resolution, output_dim):
    model = Sequential()
    model.add(Flatten(input_shape=(resolution, resolution, 1)))
    model.add(Dense(output_dim, activation='softmax'))
    return model

def plot_training_curves(history):
    """
    Plot accuracy and loss curves for training and validation sets.
    Args:
        history: a Keras History.history dictionary
    Returns:
        mpl figure.
    """
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(8,2))
    if 'acc' in history:
        ax_acc.plot(history['acc'], label='acc')
        if 'val_acc' in history:
            ax_acc.plot(history['val_acc'], label='Val acc')
        ax_acc.set_xlabel('epoch')
        ax_acc.set_ylabel('accuracy')
        ax_acc.legend(loc='upper left')
        ax_acc.set_title('Accuracy')

    ax_loss.plot(history['loss'], label='loss')
    if 'val_loss' in history:
        ax_loss.plot(history['val_loss'], label='Val loss')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.legend(loc='upper right')
    ax_loss.set_title('Loss')

    sns.despine(fig)
    return

batch_size = 128
nb_classes = 10
nb_epoch = 300
input_dim = 784
resolution = 28

model_ada = build_logistic_model(resolution, nb_classes)

model_ada.summary()

# compile the model
# first with gradient descent optimizer
model_ada.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_ada.fit(X_train_cat, y_train_cat,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=0, validation_data=(X_valid_cat, y_valid_cat))
score = model_ada.evaluate(X_valid_cat, y_valid_cat, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history)
def build_logistic_model_reg(resolution, output_dim, reg):
    model = Sequential()
    model.add(Flatten(input_shape=(resolution, resolution, 1)))
    model.add(Dense(output_dim, kernel_regularizer=reg, activation='softmax'))
    return model

reg = l1_l2(l1=0, l2=0.02)
model_ada_reg = build_logistic_model_reg(resolution, nb_classes, reg)

model_ada_reg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_ada_reg.fit(X_train_cat, y_train_cat,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=0, validation_data=(X_valid_cat, y_valid_cat))
score = model_ada_reg.evaluate(X_valid_cat, y_valid_cat, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history)
# then with stochastic gradient descent optimizer
model_sgd = build_logistic_model(resolution, nb_classes)

model_sgd.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_sgd.fit(X_train_cat, y_train_cat,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=0, validation_data=(X_valid_cat, y_valid_cat))
score = model_sgd.evaluate(X_valid_cat, y_valid_cat, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history)
def build_logisticHidden_model(resolution, output_dim, hidden_dim):
    model = Sequential()
    model.add(Flatten(input_shape=(resolution, resolution, 1)))
#     kernel_initializer='glorot_normal'
    model.add(Dense(hidden_dim[0], activation='relu'))
    model.add(Dense(hidden_dim[1], activation='relu'))
    model.add(Dense(hidden_dim[2], activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    return model

hidden_dim = [1024, 300, 50]
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_hid = build_logisticHidden_model(resolution, nb_classes, hidden_dim)

model_hid.summary()

# compile the model
model_hid.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_hid.fit(X_train_cat, y_train_cat,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=0, validation_data=(X_valid_cat, y_valid_cat))
score = model_hid.evaluate(X_valid_cat, y_valid_cat, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history)
def build_logisticHidden_model_reg(resolution, output_dim, hidden_dim, reg):
    model = Sequential()
    model.add(Flatten(input_shape=(resolution, resolution, 1)))
    model.add(Dense(hidden_dim[0], kernel_regularizer=reg, activation='relu'))
    model.add(Dense(hidden_dim[1], kernel_regularizer=reg, activation='relu'))
    model.add(Dense(hidden_dim[2], kernel_regularizer=reg, activation='relu'))
    model.add(Dense(output_dim, kernel_regularizer=reg, activation='softmax'))
    return model

model_hid_reg = build_logisticHidden_model_reg(resolution, nb_classes, hidden_dim, reg)

model_hid_reg.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_hid_reg.fit(X_train_cat, y_train_cat,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=0, validation_data=(X_valid_cat, y_valid_cat))
score = model_hid_reg.evaluate(X_valid_cat, y_valid_cat, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history)
def build_logisticHidden_model_dropOut(resolution, output_dim, hidden_dim):
    model = Sequential()
    model.add(Flatten(input_shape=(resolution, resolution, 1)))
    model.add(Dense(hidden_dim[0], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dim[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dim[2], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, activation='softmax'))
    return model

model_hid_drop = build_logisticHidden_model_dropOut(resolution, nb_classes, hidden_dim)

model_hid_drop.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_hid_drop.fit(X_train_cat, y_train_cat,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=0, validation_data=(X_valid_cat, y_valid_cat))
score = model_hid_drop.evaluate(X_valid_cat, y_valid_cat, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history)
def build_logisticHidden_model_regDrop(resolution, output_dim, hidden_dim):
    model = Sequential()
    model.add(Flatten(input_shape=(resolution, resolution, 1)))
    model.add(Dense(hidden_dim[0], kernel_regularizer=reg, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dim[1], kernel_regularizer=reg, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_dim[2],kernel_regularizer=reg,  activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, kernel_regularizer=reg, activation='softmax'))
    return model

model_hid_regDrop = build_logisticHidden_model_regDrop(resolution, nb_classes, hidden_dim)

model_hid_regDrop.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_hid_regDrop.fit(X_train_cat, y_train_cat,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=0, validation_data=(X_valid_cat, y_valid_cat))
score = model_hid_regDrop.evaluate(X_valid_cat, y_valid_cat, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history)
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = [2, 2]
# convolution kernel size
kernel_size = [3, 3]

input_shape = [img_rows, img_cols, 1]

# model layers
cnn = Sequential()
cnn.add(Conv2D(nb_filters, kernel_size, padding='same', input_shape=input_shape))
cnn.add(Activation('relu'))
cnn.add(BatchNormalization())
cnn.add(Conv2D(nb_filters, kernel_size, padding='same'))
cnn.add(Activation('relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=pool_size))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(128))
cnn.add(Activation('relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))
cnn.add(Dense(nb_classes))
cnn.add(Activation('softmax'))

cnn.summary()

cnn.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

history = cnn.fit(X_train_cat, y_train_cat,
                 batch_size=batch_size, epochs=nb_epoch,
                 verbose=0, validation_data=(X_valid_cat, y_valid_cat))
score = cnn.evaluate(X_valid_cat, y_valid_cat, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history)
# define path to save model
model_path = './cnn_notMNIST.h5'
# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_acc', 
        patience=20,
        mode='max',
        verbose=1),
    ModelCheckpoint(model_path,
        monitor='val_acc', 
        save_best_only=True, 
        mode='max',
        verbose=1),
    ReduceLROnPlateau(
        factor=0.1, 
        patience=5, 
        min_lr=0.00001, 
        verbose=1)
]

# model layers
cnn_ad = Sequential()
cnn_ad.add(Conv2D(nb_filters, kernel_size, padding='same', input_shape=input_shape))
cnn_ad.add(Activation('relu'))
cnn_ad.add(BatchNormalization())
cnn_ad.add(Conv2D(nb_filters, kernel_size, padding='same'))
cnn_ad.add(Activation('relu'))
cnn_ad.add(BatchNormalization())
cnn_ad.add(MaxPooling2D(pool_size=pool_size))
cnn_ad.add(Dropout(0.25))
cnn_ad.add(Flatten())
cnn_ad.add(Dense(128))
cnn_ad.add(Activation('relu'))
cnn_ad.add(BatchNormalization())
cnn_ad.add(Dropout(0.5))
cnn_ad.add(Dense(nb_classes))
cnn_ad.add(Activation('softmax'))

cnn_ad.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

history = cnn_ad.fit(X_train_cat, y_train_cat,
                 batch_size=batch_size, epochs=nb_epoch,
                 verbose=0, validation_data=(X_valid_cat, y_valid_cat),
                 shuffle=True, callbacks=callbacks)
score = cnn_ad.evaluate(X_valid_cat, y_valid_cat, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history)