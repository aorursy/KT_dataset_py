!pip install --upgrade wandb
import os 

import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



import tensorflow as tf

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import *

import tensorflow_addons as tfa



import wandb

from wandb.keras import WandbCallback
# Load the data

def load_data(path):



    train = pd.read_csv(path+"train.csv")

    test = pd.read_csv(path+"test.csv")

    

    x_tr = train.drop(labels=["label"], axis=1)

    y_tr = train["label"]

    

    print(f'Train: we have {x_tr.shape[0]} images with {x_tr.shape[1]} features and {y_tr.nunique()} classes')

    print(f'Test: we have {test.shape[0]} images with {test.shape[1]} features')

    

    return x_tr, y_tr, test





def seed_all(seed):

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)







def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(10,10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
DEBUG = False       # set to True in case of testing/debugging

DATA_AUGM = False   # set to True if you wish to add data augmentation 



BATCH_SIZE = 64



if DEBUG:

    EPOCHS = 3          

else: 

    EPOCHS = 40
SEED = 26

PATH = "../input/digit-recognizer/"



seed_all(SEED)



x_train, y_train, x_test = load_data(path=PATH)
sns.countplot(y_train)

for i in range(9):

    print(i, y_train[y_train==i].count().min())
# Check the data for NaNs

x_train.isnull().sum().sum(), x_test.isnull().sum().sum()   # .describe()
# Normalize the data

x_train = x_train / 255.0

x_test = x_test / 255.0
# Define image sizes and reshape to a 4-dim tensor



IMG_H, IMG_W = 28, 28

NO_CHANNELS = 1           # for greyscale images



# Reshape to a 3-dim tensor

x_train = x_train.values.reshape(-1, IMG_H, IMG_W, NO_CHANNELS)

x_test = x_test.values.reshape(-1, IMG_H, IMG_W, NO_CHANNELS)





# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=10)



print('Tensor shape (train): ', x_train.shape)

print('Tensor shape (test): ', x_test.shape)

print('Tensor shape (target ohe): ', y_train_ohe.shape)
# Split the train and the validation set for the fitting



x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train_ohe, test_size=0.15, random_state=SEED)



print('Tensors shape (train):', x_tr.shape, y_tr.shape)

print('Tensors shape (valid):', x_val.shape, y_val.shape)
# visualize some examples



plt.figure(figsize=(12, 4))

for i in range(10):  

    plt.subplot(2, 5, i+1)

    plt.imshow(x_train[i].reshape((28,28)), cmap=plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
# Build CNN model 

# CNN architecture: In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



# see original: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6 





def build_model():

    

    fs = config.filters         # 32

    k1 = config.kernel_1        #[(5,5), (3,3)]

    k2 = config.kernel_2        # [(5,5), (3,3)]

    pad = config.padding

    activ = config.activation   # 'relu'

    pool = config.pooling       # (2,2)

    dp = config.dropout         # 0.25

    dp_out = config.dropout_f   # 0.5

    dense_units = config.dense_units  # 256

    

    inp = Input(shape=(IMG_H, IMG_W, NO_CHANNELS))     # x_train.shape[1:]

    

    # layer-1:: CNN-CNN-(BN)-Pool-dp

    x = Conv2D(filters=fs, kernel_size=k1, padding=pad, activation=activ)(inp)

    x = Conv2D(filters=fs, kernel_size=k1, padding=pad, activation=activ)(x)

    x = BatchNormalization()(x)

    x = MaxPool2D(pool_size=(2,2))(x)

    x = Dropout(dp)(x)    

    

    # layer-2:: CNN-CNN-(BN)-Pool-dp

    x = Conv2D(filters=fs*2, kernel_size=k2, padding=pad, activation=activ)(inp)

    x = Conv2D(filters=fs*2, kernel_size=k2, padding=pad, activation=activ)(x)

    x = BatchNormalization()(x)

    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    x = Dropout(dp)(x)  

    

    x = Flatten()(x)

    #     x = GlobalAveragePooling2D()(x)

    

    # FC head

    x = Dense(dense_units, activation=activ)(x)

    x = Dropout(dp_out)(x)

    

    out = Dense(10, activation="softmax")(x)

    

    model = tf.keras.models.Model(inp, out)

    

    print(model.summary())

    return model



# you may also experiment with this arhitecture (see credits)

# https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist/notebook#Train-15-CNNs



def build_lenet():

    

    fs = config.filters         # 32

    #     k1 = config.kernel_1        #[(5,5), (3,3)]

    #     k2 = config.kernel_2        # [(5,5), (3,3)]

    #     pad = config.padding

    activ = config.activation   # 'relu'

    dp = config.dropout         # 0.25

    

    

    inp = Input(shape=(28, 28, 1)) 

    

    

    

    x = Conv2D(fs, kernel_size = 3, activation=activ)(inp)

    x = BatchNormalization()(x)

    x = Conv2D(fs, kernel_size = 3, activation=activ)(x)

    x = BatchNormalization()(x)

    x = Conv2D(fs, kernel_size = 5, strides=2, padding='same', activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.4)(x)

    

    x = Conv2D(fs*2, kernel_size = 3, activation=activ)(x)

    x = BatchNormalization()(x)

    x = Conv2D(fs*2, kernel_size = 3, activation=activ)(x)

    x = BatchNormalization()(x)

    x = Conv2D(fs*2, kernel_size = 5, strides=2, padding='same', activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.4)(x)

    

    x = Conv2D(128, kernel_size = 4, activation=activ)(x)

    x = BatchNormalization()(x)

    x = Flatten()(x)

    x = Dropout(0.4)(x)

    out = Dense(10, activation='softmax')(x)

    

    model = tf.keras.models.Model(inp, out)



    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST

    #     model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    

    return model
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

api_key = user_secrets.get_secret("API_key")
!wandb login $api_key
hyperparams = dict(

    filters = 64,

    kernel_1 = (5,5),

    kernel_2 = (3,3),

    padding = 'same',

    pooling = (2,2),

    lr = 0.001,

    wd = 0.0,

    lr_schedule = 'RLR',    # cos, cyclic, step decay

    optimizer = 'Adam+SWA',     # RMS

    dense_units = 256,

    activation = 'elu',      # elu, LeakyRelu

    dropout = 0.25,

    dropout_f = 0.5,

    batch_size = BATCH_SIZE,

    epochs = EPOCHS,

)
wandb.init(project="kaggle-mnist", config=hyperparams)

config = wandb.config
config.keys()
model = build_model()
# Define the optimizer



LR = config.lr     # 0.001



if config.optimizer=='Adam':

    opt = Adam(LR)

elif config.optimizer=='RMS':

    opt = RMSprop(lr=LR, rho=0.9, epsilon=1e-08, decay=0.0)

elif config.optimizer=='Adam+SWA':

    opt = Adam(LR)

    opt = tfa.optimizers.SWA(opt)

else: 

    opt = 'adam'    # native adam optimizer 

    

    

# Compile the model

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
# mdl_path = '../input/output/'



callbacks = [

    EarlyStopping(monitor='val_accuracy', patience=15, verbose=1),

    #    ModelCheckpoint(mdl_path+'best_model.h5')

    ReduceLROnPlateau(monitor='val_accuracy', patience=10, verbose=1, factor=0.5, min_lr=1e-4), 

    WandbCallback(monitor='val_loss')

]
if DATA_AUGM:

    

    # copied data generator 

    

    datagen = ImageDataGenerator(

        featurewise_center=False,                  # set input mean to 0 over the dataset

        samplewise_center=False,                   # set each sample mean to 0

        featurewise_std_normalization=False,       # divide inputs by std of the dataset

        samplewise_std_normalization=False,        # divide each input by its std

        zca_whitening=False,                       # apply ZCA whitening

        rotation_range=10,                         #  randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1,                          #  Randomly zoom image 

        width_shift_range=0.1,                     # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,                    # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,                     # randomly flip images

        vertical_flip=False)                       # randomly flip images



    datagen.fit(x_train)
# # Fit the model



# if DATA_AUGM:

#     # With data augmentation to prevent overfitting (accuracy: 0.99286)

#     hist = model.fit_generator(datagen.flow(x_train, y_train_ohe, batch_size=BATCH_SIZE),

#                                epochs=EPOCHS, 

#                                validation_data=(x_val, y_val),

#                                verbose=1, 

#                                steps_per_epoch=x_train.shape[0] // BATCH_SIZE, 

#                                callbacks=callbacks)

    

# else: 

#     # Without data augmentation (accuracy: 0.98114)

#     hist = model.fit(x_train, y_train_ohe, 

#                      batch_size=config.batch_size,    # BATCH_SIZE, 

#                      epochs=config.epochs,            # EPOCHS, 

#                      validation_data=(x_val, y_val), 

#                      callbacks=callbacks,

#                      verbose=1) 
hist = model.fit(x_train, y_train_ohe, 

                     batch_size=config.batch_size,    # BATCH_SIZE, 

                     epochs=config.epochs,            # EPOCHS, 

                     validation_data=(x_val, y_val), 

                     callbacks=callbacks,

                     verbose=1) 
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(1,2, figsize=(12, 6))

ax[0].plot(hist.history['loss'], color='b', label="Training loss")

ax[0].plot(hist.history['val_loss'], color='r', label="Validation loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(hist.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(hist.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Predict the values from the validation dataset

Y_pred = model.predict(x_val)



# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred, axis=1) 



# Convert validation observations to one hot vectors

Y_true = np.argmax(y_val, axis=1) 



# compute the confusion matrix

cm = confusion_matrix(Y_true, Y_pred_classes) 



# plot the confusion matrix

plot_confusion_matrix(cm, classes=range(10)) 
# Confusion Matrix

wandb.sklearn.plot_confusion_matrix(Y_true, Y_pred_classes, labels=range(10))
# predict results

results = model.predict(x_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist.csv",index=False)