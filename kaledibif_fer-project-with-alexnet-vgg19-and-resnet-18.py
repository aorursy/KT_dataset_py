import os
import numpy as np 
import pandas as pd 
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import itertools
import pickle
import random

from PIL import Image
from scipy import interp
from random import randint
from sklearn import metrics, decomposition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from keras import backend as K
from keras import callbacks
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Activation, Dropout, Flatten, BatchNormalization, Input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
from keras.models import model_from_json, Model, Sequential
from keras.optimizers import *

from keras.preprocessing import image
from keras.applications import VGG19
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.regularizers import l2
from keras.utils.data_utils import Sequence
# detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

simple_model_epochs = 40
simple_model_bs = 32

resnet_model_epochs = 16
resnet_model_bs = 64

vgg_model_epochs = 50
vgg_model_bs = 16
filname = '../input/fer2013/fer2013.csv'
names = ['emotion','pixels','usage']
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
num_classes = len(label_map)

df = pd.read_csv('../input/fer2013/fer2013.csv', names=names, na_filter=False)
im = df['pixels']

def getData(filname):
    Y = []
    X = []
    first = True
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

X, Y = getData(filname)
num_class = len(set(Y))
X = X * 255

N, D = X.shape
X = X.reshape(N, 48, 48, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

y_train_flat = y_train
y_test_flat = y_test

y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
fig = plt.figure(figsize = (15, 10))

for counter, img in enumerate(X_train[:12]):
    ax = fig.add_subplot(3, 4, counter + 1)
    ax.imshow(np.asarray(X_train[counter]).reshape(48, 48), cmap = 'gray')
    plt.title(label_map[Y[counter]])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

plt.show()
datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

datagen.fit(X_train)
def plot_confusion_matrix(y_test, 
                          y_pred,
                          title='Unnormalized confusion matrix'):
    
    classes=np.array(("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"))
    
    cmap=plt.cm.Blues
    cm = confusion_matrix(y_test, y_pred)
    
    # if normalize:
        # cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        
    np.set_printoptions(precision=2)
        
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.min() + (cm.max() - cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True expression')
    plt.xlabel('Predicted expression')
    plt.show()
    plt.savefig(title + '.png')
    
def randomColorGenerator(number_of_colors = 1, seed = 0):
    '''Generate list of random colors'''
    np.random.seed(seed)
    return ["#"+''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)]) for i in range(number_of_colors)]

def make_fpr_tpr_auc_dicts(y, probs_list):
    '''Compute and return the ROC curve and ROC area for each class in dictionaries'''
    # Dicts
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    
    # For test
    for i in range(num_classes):
        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y[:, i], probs_list[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y.ravel(), probs_list.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, thresholds, roc_auc

def plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(-0.0025, 0.03), ylim=(0.99, 1.001), seed=0, save_title=None):
    '''Plot ROC AUC Curves'''
    fig, axes = plt.subplots(nrows=1, ncols=2, dpi=150, figsize=(10,5))
    
    lw = 2
    axes[0].set_xlabel('False Positive Rate')
    axes[1].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    
    if num_classes!=4:
        class_colors = randomColorGenerator(num_classes, seed)
    
    for i in range(num_classes):
        axes[0].plot(fpr[i], tpr[i], color=class_colors[i], label='{0} ({1:0.2f}%)' ''.format(label_map[i], roc_auc[i]*100))
        axes[1].plot(fpr[i], tpr[i], color=class_colors[i], lw=lw, label='{0} ({1:0.2f}%)' ''.format(label_map[i], roc_auc[i]*100))
    
    axes[0].plot(fpr['micro'], tpr['micro'], label='Micro avg ({:0.2f}%)' ''.format(roc_auc['micro']*100), linestyle=':', color='deeppink')
    axes[0].plot(fpr['macro'], tpr['macro'], label='Macro avg ({:0.2f}%)' ''.format(roc_auc['macro']*100), linestyle=':', color='navy')
    axes[0].plot([0, 1], [0, 1], color='k', linestyle='--', lw=0.5)
    axes[0].scatter(0,1, label='Ideal', s=2)
    
    axes[1].plot(fpr['micro'], tpr['micro'], lw=lw, label='Micro avg ({:0.2f}%)'.format(roc_auc['micro']*100), linestyle=':', color='deeppink')
    axes[1].plot(fpr['macro'], tpr['macro'], lw=lw, label='Macro avg ({:0.2f}%)'.format(roc_auc['macro']*100), linestyle=':', color='navy')
    axes[1].plot([0, 1], [0, 1], color='k', linestyle='--', lw=0.5)
    axes[1].scatter(0,1, label='Ideal', s=50)
    
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    
    axes[0].grid(True, linestyle='dotted', alpha=1)
    axes[1].grid(True, linestyle='dotted', alpha=1)
    
    axes[0].legend(loc=4)
    axes[1].legend(loc=4)
    
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(f'{save_title}.pdf', bbox_inches='tight', format='pdf', dpi=200)
    plt.show()
### SIMPLE MODEL

def simple_model():
    input_shape = (48,48,1)
    simple_model = Sequential()
    simple_model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
    simple_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    simple_model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))
    simple_model.add(Activation('relu'))
    simple_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    simple_model.add(Conv2D(64, (3, 3), activation = 'relu'))
    simple_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    simple_model.add(Flatten())
    simple_model.add(Dense(128, activation = 'relu'))
    simple_model.add(Dropout(0.5))
    
    simple_model.add(Dense(7, activation = 'softmax'))
    
    simple_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='RMSprop')

    return simple_model
### RESNET MODEL

subtract_pixel_mean = True
n = 3
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

batch_size = 16
epochs = 20
data_augmentation = True
num_classes = 7
input_shape = X_train.shape[1:]

def resnet_layer(inputs,num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=7):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
   
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(y)
    
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

resnet_model = resnet_v1(input_shape=input_shape,depth=depth)
optimizer = Adam(lr=0.0001)

resnet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# resnet_model.summary()
### SIMPLE MODEL - FIT

path_model='/kaggle/working/simple_model.h5'
simple_model=simple_model()

# K.tensorflow_backend.clear_session()
# with tpu_strategy.scope():
simple_model_result = simple_model.fit_generator(datagen.flow(X_train, y_train, batch_size = simple_model_bs),
                                                     validation_data = (X_test, y_test),
                                                     epochs = simple_model_epochs,
                                                     # steps_per_epoch = 4096,
                                                     verbose = 1,
                                                     callbacks = [ModelCheckpoint(filepath = path_model)])

# Save trained model to disk
filename = 'trained_simple_model.sav'
pickle.dump(simple_model_result, open(filename, 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))

simple_model_scores = simple_model.evaluate(X_test, y_test, verbose=1)
print('Simple Model Test Loss:', simple_model_scores[0])
print('Simple Model Test Accuracy:', simple_model_scores[1])
rnd = random.randint(0, X_train.shape[0])
rnd = 2
s1 = np.array(X_train[rnd].reshape(48, 48))
img = Image.fromarray(s1)
img = img.resize((400,400))

plt.title(label_map[Y[rnd]])
plt.imshow(img)
### SIMPLE MODEL - PLOTS

plt.plot(simple_model_result.history['accuracy'])
plt.plot(simple_model_result.history['val_accuracy'])
plt.title('Simpe Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('Simpe Model Accuracy.png')

plt.plot(simple_model_result.history['loss'])
plt.plot(simple_model_result.history['val_loss'])
plt.title('Simpe Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('Simpe Model Loss.png')

simple_model_y_pred_ = simple_model.predict(X_test, verbose=1)
simple_model_y_pred = np.argmax(simple_model_y_pred_, axis=1)
simple_model_t_te = np.argmax(y_test, axis=1)
### SIMPLE MODEL - PLOTS

fig = plot_confusion_matrix(y_test=simple_model_t_te,
                            y_pred=simple_model_y_pred,
                            title='Simple Model Average Accuracy: ' + str(round(np.sum(simple_model_y_pred == simple_model_t_te)/len(simple_model_t_te), 2)) + '\n')
### SIMPLE MODEL - PLOTS

fpr, tpr, thresholds, roc_auc = make_fpr_tpr_auc_dicts(y_test, simple_model_y_pred_)
plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(0, 0.2), ylim=(0.8, 1), seed=5, save_title='Simple: ROC')
#aug_train = ImageDataGenerator(fill_mode="nearest")
#aug_train.fit(X_train)

#generator_val = ImageDataGenerator()
#generator_val.fit(X_test)

vgg_conv = VGG19(weights=None, include_top=False, input_shape=(48, 48,1))

vgg_model = Sequential()
vgg_model.add(vgg_conv)

vgg_model.add(Flatten())
vgg_model.add(Dense(7,  kernel_initializer='normal', activation='softmax'))
vgg_model.compile(loss='mean_squared_error', optimizer=Adadelta(), metrics=['accuracy'])

aug = ImageDataGenerator(rotation_range=25,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

filename='model_train_new.csv'
filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log,checkpoint]
callbacks_list = [csv_log]

# with tpu_strategy.scope():
vgg_model_result = model.fit_generator(aug.flow(X_train, y_train, batch_size=vgg_model_bs),
                                       validation_data=(X_test, y_test),
                                       steps_per_epoch=len(X_train) // vgg_model_bs,
                                       epochs=vgg_model_epochs,
                                       verbose=1,
                                       callbacks = callbacks_list)

vgg_model_scores = vgg_model.evaluate(X_test, y_test, verbose=1)
print('VGG Model Test Loss:', vgg_model_scores[0])
print('VGG Model Test Accuracy:', vgg_model_scores[1])
### VGG MODEL - PLOTS

plt.plot(vgg_model_result.history['accuracy'])
plt.plot(vgg_model_result.history['val_accuracy'])
plt.title('VGG19 Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('VGG19 Model Loss.png')

plt.plot(vgg_model_result.history['loss'])
plt.plot(vgg_model_result.history['val_loss'])
plt.title('VGG19 Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('VGG19 Model Loss.png')
### VGG MODEL - PLOTS

vgg_model_y_pred_ = vgg_model.predict(X_test, verbose=1)
vgg_model_y_pred = np.argmax(vgg_model_y_pred_, axis=1)
vgg_model_t_te = np.argmax(y_test, axis=1)

fig = plot_confusion_matrix(y_test=vgg_model_t_te, y_pred=vgg_model_y_pred,
                            title='VGG19 Average Accuracy: ' + str(round(np.sum(vgg_model_y_pred == vgg_model_t_te)/len(vgg_model_t_te), 2)) + '\n')
### VGG MODEL - PLOTS

fpr, tpr, thresholds, roc_auc = make_fpr_tpr_auc_dicts(y_test, vgg_model_y_pred_)
plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(0, 0.2), ylim=(0.8, 1), seed=5, save_title='ROC: VGG19')
### RESNET MODEL - FIT

filepath = '/kaggle/working/resnet_model.h5'

# with tpu_strategy.scope():
resnet_model_result = resnet_model.fit_generator(datagen.flow(X_train, y_train, batch_size=256), 
                                                 steps_per_epoch=1024,
                                                 validation_data=(X_test, y_test),
                                                 epochs=20,
                                                 verbose=1,
                                                 workers=4,
                                                 callbacks=[ModelCheckpoint(filepath=filepath)])

# Save trained model to disk
filename = 'trained_resnet_model.sav'
pickle.dump(resnet_model_result, open(filename, 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))

resnet_model_scores = resnet_model.evaluate(X_test, y_test, verbose=1)
print('ResNet Model Test Loss:', resnet_model_scores[0])
print('ResNet Model Test Accuracy:', resnet_model_scores[1])
### SIMPLE MODEL - PLOTS

plt.plot(resnet_model_result.history['accuracy'])
plt.plot(resnet_model_result.history['val_accuracy'])
plt.title('ResNet Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('ResNet Model Loss.png')

plt.plot(resnet_model_result.history['loss'])
plt.plot(resnet_model_result.history['val_loss'])
plt.title('ResNet Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('ResNet Model Loss.png')
### SIMPLE MODEL - PLOTS

resnet_model_y_pred_ = resnet_model.predict(X_test, verbose=1)
resnet_model_y_pred = np.argmax(resnet_model_y_pred_, axis=1)
resnet_model_t_te = np.argmax(y_test, axis=1)

fig = plot_confusion_matrix(y_test=resnet_model_t_te, y_pred=resnet_model_y_pred,
                            title='ResNet Average Accuracy: ' + str(round(np.sum(resnet_model_y_pred == resnet_model_t_te)/len(resnet_model_t_te), 2)) + '\n')
### SIMPLE MODEL - PLOTS

fpr, tpr, thresholds, roc_auc = make_fpr_tpr_auc_dicts(y_test, resnet_model_y_pred_)
plot_roc_auc_curves(fpr, tpr, roc_auc, xlim=(0, 0.2), ylim=(0.8, 1), seed=5, save_title='ROC: ResNet')