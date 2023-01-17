import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

import os

import cv2

import tensorflow as tf



import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = [16, 8]



print('Using Tensorflow version:', tf.__version__)
try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
!wget https://github.com/kittinan/thai-handwriting-number/raw/master/src/thainumber_28.pkl
import pickle

data = pickle.load(open('thainumber_28.pkl','rb'))

X = data['X']

Y = data['Y'].astype(int)
print(X.shape)

print(len(np.unique(Y)))
(unique, counts) = np.unique(Y, return_counts=True)

unique, counts
n = 10

for j, i in enumerate(np.random.choice(len(X), n)):

  plt.subplot(1,n,j+1)

  plt.imshow(X[i,:,:,0])

  plt.axis('off')

plt.show()
IMG_SIZE_h = 32

IMG_SIZE_w = 32

BATCH_SIZE = 64

LR = 0.001

EPOCHS = 100

WARMUP = 20

AUTO = tf.data.experimental.AUTOTUNE

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
def decode_image(image, label=None, image_size=(IMG_SIZE_h, IMG_SIZE_w)):

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

    if label is None:

        return image

    else:

        label = to_categorical(label)

        return image, label
X_train,Y_train = decode_image(X,Y)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, y_train = decode_image(X_train,y_train)

X_val, y_val = decode_image(X_val,y_val)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
from tensorflow.keras.optimizers import RMSprop



optimizer = RMSprop(lr=LR, rho=0.9, epsilon=1e-08, decay=0.0)
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
with strategy.scope():

    model = Sequential(name="THAI_MNIST")



    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (IMG_SIZE_h,IMG_SIZE_w,1), name = "first_cov"))

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu',name = "second_cov"))

    model.add(MaxPool2D(pool_size=(2,2),name = "first_MaxPool"))

    model.add(Dropout(0.25,name = "first_DropOut"))



    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu', name="third_cov"))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu', name="forth_cov"))

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2),name = "second_MaxPool"))

    model.add(Dropout(0.25,name = "second_DropOut"))





    model.add(Flatten(name = "first_Flatten"))

    model.add(Dense(256, activation = "relu",name = "first_Dense"))

    model.add(Dropout(0.5,name = "thrid_DropOut"))

    model.add(Dense(10, activation = "softmax",name = "second_Dense"))

    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    model.summary()
from tensorflow.keras.callbacks import EarlyStopping

import math



def get_cosine_schedule_with_warmup(lr, num_warmup_steps, num_training_steps, num_cycles=0.5):

    def lrfn(epoch):

        if epoch < num_warmup_steps:

            return float(epoch) / float(max(1, num_warmup_steps)) * lr

        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr



    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



lr_schedule= get_cosine_schedule_with_warmup(lr=LR, num_warmup_steps=WARMUP, num_training_steps=EPOCHS)





es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),

                              epochs = EPOCHS, 

                              validation_data = (X_val,y_val),

                              verbose = 1, 

                              steps_per_epoch=X_train.shape[0] // BATCH_SIZE, 

                              callbacks=[lr_schedule,es])
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
from sklearn.metrics import confusion_matrix





def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i in range (cm.shape[0]):

        for j in range (cm.shape[1]):

            plt.text(j, i, cm[i, j],

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error][:,:,0]))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
# https://keras.io/examples/vision/grad_cam/

from tensorflow import keras



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):

    last_conv_layer = model.get_layer(last_conv_layer_name)

    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)



    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input

    for layer_name in classifier_layer_names:

        x = model.get_layer(layer_name)(x)

    classifier_model = keras.Model(classifier_input, x)



    with tf.GradientTape() as tape:

        last_conv_layer_output = last_conv_layer_model(img_array)

        tape.watch(last_conv_layer_output)

        preds = classifier_model(last_conv_layer_output)

        top_pred_index = tf.argmax(preds[0])

        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))



    last_conv_layer_output = last_conv_layer_output.numpy()[0]

    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):

        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)



    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap, top_pred_index.numpy()
def superimposed_img(image, heatmap):

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")



    jet_colors = jet(np.arange(256))[:, :3]

    jet_heatmap = jet_colors[heatmap]



    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)

    jet_heatmap = jet_heatmap.resize((32, 32))

    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)



    superimposed_img = jet_heatmap * 0.4 + image

    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img
test_image = X_val[1][:,:,0]

plt.imshow(test_image)

test_image = np.expand_dims(test_image,axis=0)
last_conv_layer_name = "second_MaxPool"

classifier_layer_names = [

    "second_DropOut",

    "first_Flatten",

    "first_Dense",

    "thrid_DropOut",

    "second_Dense",

]
heatmap, top_index = make_gradcam_heatmap(test_image, model, last_conv_layer_name, classifier_layer_names)

print("predicted as", top_index)
stacked_img = np.stack((test_image[0],)*3, axis=-1)
s_img = superimposed_img((stacked_img), heatmap)

plt.imshow(s_img)