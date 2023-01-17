import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Model
from keras.layers import (Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation,
                          BatchNormalization, Concatenate)
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import os, cv2, random, re, csv
from tqdm import tqdm

import time
t_start = time.time()
TRAIN_DIR = '../input/train'
TEST_DIR = '../input/test'

# Initial size is 224
ROWS = 224
COLS = 224
CHANNELS = 1

# because of the limited resources we have, we have to adapt the BATCH_SIZE 
# With image size and complexity of the model (nb params)
BATCH_SIZE=40
EPOCHS=40

kfold_splits = 3
# Separating cats and dogs for exploratory analysis

train_images = [TRAIN_DIR+"/"+i for i in os.listdir(TRAIN_DIR)]
train_dogs = [TRAIN_DIR+"/"+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [TRAIN_DIR+"/"+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images = [TEST_DIR+"/"+i for i in os.listdir(TEST_DIR)]

#### For testing purposes
train_images = train_dogs[:4000] + train_cats[:4000]
test_images = test_images[:1000]
random.shuffle(train_images)

def read_image(file_path):
    img= cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    X = [] # images as arrays
    y = [] # labels
    for image_file in tqdm(images):
        try:
            image = read_image(image_file)
            X.append(image)
            if 'dog' in image_file: y.append(1)
            elif 'cat' in image_file: y.append(0)
        except:
            pass #print(image_file)
    X = np.array(X)
    X = np.expand_dims(X, axis=3)
    y = np.array(y)
    return X, y

print("Processing Train images")
X_train, y_train = prep_data(train_images)

print("Train: {} images with shape {}".format(len(X_train),X_train[0].shape))
print("Test: {} images".format(len(test_images)))
# We're dealing with classification problem here - (1) dogs (0) cats
labels = [1 if 'dog' in l else 0 for l in train_images]
sns.countplot(labels)
plt.title('Cats and Dogs');
# A quick side-by-side comparison of the animals
for idx in range(2):
    idx = idx + np.random.randint(low=1, high=100); # To randomize images
    cat = read_image(train_cats[idx])
    dog = read_image(train_dogs[idx])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(15, 5))
    f = plt.imshow(pair)
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)
    plt.show()
def convBatchActivMax_block(_input, N_Filters, N, kernel, blockNumber):
    # N is used to Variate number of filters for each block
    x = Conv2D(N_Filters* N, kernel_size=kernel, padding='same', activation='relu', name='block{}_conv{}_{}'.format(blockNumber, 1, kernel))(_input)
    x = Conv2D(N_Filters* N, kernel_size=kernel, padding='same', name='block{}_conv{}_{}'.format(blockNumber, 2, kernel))(x)
    x = BatchNormalization(name="block{}_BatchNorm_{}".format(blockNumber,kernel))(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((2,2), strides=(2,2), name='block{}_pool_{}'.format(blockNumber, kernel))(x)
    return x

def build_model(N_Filters=32):
    input_layer = Input((ROWS, COLS, CHANNELS), name="InputLayer")
    
    #----- Branch 1-------
    ######################
    # Block 1
    x1 = convBatchActivMax_block(input_layer, N_Filters, 1, 3, 1)
    # Block 2
    x1 = convBatchActivMax_block(x1, N_Filters, 2, 3, 2)
    # Block 3
    x1 = convBatchActivMax_block(x1, N_Filters, 3, 3, 3)
    
    #----- Branch 2-------
    ######################
    # Block 1
    x2 = convBatchActivMax_block(input_layer, N_Filters, 1, 5, 1)
    # Block 2
    x2 = convBatchActivMax_block(x2, N_Filters, 2, 5, 2)
    # Block 3
    x2 = convBatchActivMax_block(x2, N_Filters, 3, 5, 3)
    
    OutConcat = Concatenate()([x1,x2])
    x = Conv2D(N_Filters*3, 1, activation='relu')(OutConcat)
    
    x = Flatten(name='flatten')(x)
    x = Dense(N_Filters*10, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(N_Filters*4, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input_layer, output)
    model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model
model = build_model()
model.summary()
#################
# Plot The Model
#################
from keras.utils import plot_model 
plot_model(model, to_file='keras-baseline-architecture.png')

from IPython.display import Image
Image(filename='keras-baseline-architecture.png') 
###############################
# Train & Validation Generators
###############################
# Augmentation configuration to use for training and validation
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)
#############
# Callbacks:
############
def get_callbacks(name_weights, patience_lr):
    erl_stop = EarlyStopping(monitor='val_loss', mode = 'min',patience=patience_lr*2, verbose=1)
    mcp_save = ModelCheckpoint(name_weights, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=patience_lr, min_lr=0.0001, verbose=1)
    return [erl_stop, mcp_save, reduce_lr_loss]
from sklearn.model_selection import StratifiedKFold

# Instantiate the cross validator
skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)
hists = []
# Loop through the indices the split() method returns
for index, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print("___________________________________________")
    print("Training on fold (" + str(index+1) + "/" + str(kfold_splits) + ")")
    print("___________________________________________")
    
    # Generate batches from indices
    xtrain = X_train[train_idx]
    xval = X_train[val_idx] / 255
    ytrain = y_train[train_idx]
    yval = y_train[val_idx]
    
    name_weights = "final_model_fold" + str(index+1) + "_weights.h5"
    Callbacks = get_callbacks(name_weights = name_weights, patience_lr=3)
    
    # Prepare generators for training and validation sets
    train_generator = train_datagen.flow(xtrain, ytrain, batch_size=BATCH_SIZE)

    # Clear model, and create it
    model = None
    model = build_model()
    history = model.fit_generator(
        train_generator, 
        steps_per_epoch = len(xtrain) // BATCH_SIZE,
        callbacks = Callbacks,
        epochs = EPOCHS,
        shuffle=True,
        validation_data = [xval, yval],
        validation_steps = len(xval) // BATCH_SIZE,
        verbose = 1
    )
    
    hists.append(history)
    accuracy_history = history.history['loss']
    val_accuracy_history = history.history['val_loss']
    print("\nLast training Loss: ", str(accuracy_history[-1]), ", last validation accuracy: ", str(val_accuracy_history[-1]))
    
    # Debug message
    print("\nTraining new iteration on ", str(xtrain.shape[0]), " training samples, ", str(xval.shape[0]), " validation samples\n")
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

def plot_accuracy_and_loss(idx, hist):
    plt.plot(hists[idx].history['loss'], label='Train Loss Fold {}'.format(idx+1), color=colors[idx])
    plt.plot(hists[idx].history['val_loss'], label='Val Loss Fold {}'.format(idx+1), color=colors[idx], linestyle = "dashdot")

plt.figure(figsize=(22, 10))
plt.title('Train Accuracy vs Val Accuracy')
for idx, hist in enumerate(hists):
    plot_accuracy_and_loss(idx, hist)

plt.legend()
plt.show()
t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")
