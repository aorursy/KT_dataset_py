#

# Notebook configuration

#



DEVICE            = "TPU"  # Any value in ["TPU", "GPU"]

SEED              = 8080

FOLDS             = 5

FOLD_WEIGHTS      = [1./FOLDS]*FOLDS

BATCH_SIZE        = 256

EPOCHS            = 5000

MONITOR           = "val_loss"

MONITOR_MODE      = "min"

ES_PATIENCE       = 5

LR_PATIENCE       = 0

LR_FACTOR         = 0.5

EFF_NET           = 3

EFF_NET_WEIGHTS   = 'noisy-student'

LABEL_SMOOTHING   = 0.1

VERBOSE           = 1
!pip install -q efficientnet >> /dev/null
import numpy as np

import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras.backend as K

import efficientnet.tfkeras as efn

from sklearn.model_selection import KFold

from keras.preprocessing.image import ImageDataGenerator
if DEVICE == "TPU":

    print("connecting to TPU...")

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

    except ValueError:

        print("Could not connect to TPU")

        tpu = None



    if tpu:

        try:

            print("initializing  TPU ...")

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print("TPU initialized")

        except _:

            print("failed to initialize TPU")

    else:

        DEVICE = "GPU"



if DEVICE != "TPU":

    print("Using default strategy for CPU and single GPU")

    strategy = tf.distribute.get_strategy()



if DEVICE == "GPU":

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    



AUTO     = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

print(f'REPLICAS: {REPLICAS}')
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train.describe()
# Getting X data

X = train.drop(labels=['label'], axis=1)

X = X.astype('float32')



# Data normalization

X = X / 255



# 2D, 1 channel reshape

X = X.values.reshape(X.shape[0],28,28,1)



# Padding image so it fits on EfficientNet minimum size (32x32)

X = np.pad(X, ((0,0), (2,2), (2,2), (0,0)), mode='constant')



# Copying channel so it becomes a 3-channel image (32x32x3), required by EfficientNet

X = np.squeeze(X, axis=-1)

X = stacked_img = np.stack((X,)*3, axis=-1)



# Checking shape, it must be (n_images, 32, 32, 3)

X.shape
# Getting labels

y = train['label'].values.astype('float32')



# One-hot encoding labels

y = tf.keras.utils.to_categorical(y, 10)



y
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test.describe()
# Getting X data

X_test = test.astype('float32')



# Data normalization

X_test = X_test / 255



# 2D, 1 channel reshape

X_test = X_test.values.reshape(X_test.shape[0],28,28,1)



# Padding image so it fits on EfficientNet minimum size (32x32)

X_test = np.pad(X_test, ((0,0), (2,2), (2,2), (0,0)), mode='constant')



# Copying channel so it becomes a 3-channel image (32x32x3), required by EfficientNet

X_test = np.squeeze(X_test, axis=-1)

X_test = stacked_img = np.stack((X_test,)*3, axis=-1)



# Checking shape, it must be (n_images, 32, 32, 3)

X_test.shape
# Got this from https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

datagen = ImageDataGenerator(

    featurewise_center=False,              # set input mean to 0 over the dataset

    samplewise_center=False,               # set each sample mean to 0

    featurewise_std_normalization=False,   # divide inputs by std of the dataset

    samplewise_std_normalization=False,    # divide each input by its std

    zca_whitening=False,                   # apply ZCA whitening

    rotation_range=10,                     # randomly rotate images in the range (degrees, 0 to 180)

    zoom_range = 0.1,                      # Randomly zoom image 

    width_shift_range=0.1,                 # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,                # randomly shift images vertically (fraction of total height)

    horizontal_flip=False,                 # randomly flip images

    vertical_flip=False                    # randomly flip images

)
eff_nets = [

    efn.EfficientNetB0,

    efn.EfficientNetB1,

    efn.EfficientNetB2,

    efn.EfficientNetB3,

    efn.EfficientNetB4,

    efn.EfficientNetB5,

    efn.EfficientNetB6,

    efn.EfficientNetB7,

    efn.EfficientNetL2,

]



def build_model ():

    inp = tf.keras.layers.Input(shape=(X.shape[1], X.shape[2], X.shape[3]))

    oup = eff_nets[EFF_NET](

        input_shape=(X.shape[1], X.shape[2], X.shape[3]),

        weights=EFF_NET_WEIGHTS,

        include_top=False,

    )(inp)

    oup = tf.keras.layers.GlobalAveragePooling2D()(oup)

    oup = tf.keras.layers.Dense(512, activation='linear')(oup)

    oup = tf.keras.layers.Activation('relu')(oup)

    oup = tf.keras.layers.Dropout(0.5)(oup)

    oup = tf.keras.layers.Dense(10, activation='linear')(oup)

    oup = tf.keras.layers.Activation('softmax')(oup)

    

    model = tf.keras.Model (inputs=[inp], outputs=[oup])

    

    loss = tf.keras.losses.CategoricalCrossentropy(

        from_logits=False,

        label_smoothing=LABEL_SMOOTHING,

    )

    

    opt = tf.keras.optimizers.Nadam(learning_rate=3e-4)

    

    model.compile(optimizer=opt,loss=loss,metrics=['acc'])

    

    return model



build_model().summary()
%%time



# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit



oof   = np.zeros((X.shape[0], y.shape[1]))

preds = np.zeros((X_test.shape[0], y.shape[1]))



skf = KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)



for fold,(idxT,idxV) in enumerate(skf.split(X)):



    if DEVICE=='TPU':

        if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)

            

    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    K.clear_session()

    with strategy.scope():

        model = build_model()

        

    weights_filename='fold-%i.h5'%fold

        

    # Save best model for each fold

    sv = tf.keras.callbacks.ModelCheckpoint(

        weights_filename, monitor=MONITOR, verbose=VERBOSE, save_best_only=True,

        save_weights_only=True, mode=MONITOR_MODE, save_freq='epoch')



    # Learning rate reduction

    lrr = tf.keras.callbacks.ReduceLROnPlateau(

        monitor=MONITOR,

        factor=LR_FACTOR,

        patience=LR_PATIENCE,

        verbose=VERBOSE,

        mode=MONITOR_MODE

    )

    

    # Early stopping

    es = tf.keras.callbacks.EarlyStopping(

        monitor=MONITOR,

        patience=ES_PATIENCE,

        verbose=VERBOSE,

        mode=MONITOR_MODE,

    )

    

    # Datagen workaround

    print ('Generating train data...')

    i = 0

    datagen.fit(X[idxT])

    steps = 2 * (X[idxT].shape[0] // BATCH_SIZE)

    X_train = None

    y_train = None

    with tqdm(total=steps) as pbar:

        for arr in datagen.flow(X[idxT], y[idxT], batch_size=BATCH_SIZE):

            if X_train is None:

                X_train = arr[0]

                y_train = arr[1]

            else:

                X_train = np.concatenate((X_train, arr[0]))

                y_train = np.concatenate((y_train, arr[1]))

            i += 1

            pbar.update(1)

            if i >= steps:

                break

                

#     # Class weights

#     w = 1 / np.sum(y_train, axis=0)

#     w *= 1 / np.min(w)

#     cw = {

#         np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) : w[0]

#     }

    

    # Train

    print('Training...')

    history = model.fit(

        X_train,

        y_train,

        batch_size=BATCH_SIZE,

        epochs=EPOCHS,

        callbacks = [

            sv,

            lrr,

            es,

        ],

        validation_data=(

            X[idxV], y[idxV]

        ),

        verbose=VERBOSE,

#         class_weight = { 0: 1, 1: 56 }

    )



    print('Loading best model...')

    model.load_weights('fold-%i.h5'%fold)

    

    print('Predicting OOF...')

    oof[idxV,] = model.predict([X[idxV]],verbose=VERBOSE)

    

    print('Predicting Test...')

    preds += (model.predict([X_test], verbose=VERBOSE) * FOLD_WEIGHTS[fold])



    acc_sum = 0

    for k in idxV:

        if np.argmax(oof[k]) == np.argmax(y[k]):

            acc_sum += 1

    print('>>>> FOLD {} Accuracy = {:.4f}%'.format(fold+1,acc_sum/len(idxV)*100))

    print()
# Getting predictions output on the demanded format

final_predictions = pd.Series(np.argmax(preds, axis=1), name="Label")



# Generating dataframe on the demanded format

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),final_predictions], axis=1)



# Saving file

submission.to_csv("submission.csv",index=False)



# Printing few samples

submission.head()