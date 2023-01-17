SAMPLE_SIZE = None
TEST_SIZE = 0.1
RANDOM_STATE = 2018
BATCH_SIZE = 64
EPOCHS = 100
IMG_SIZE = 100
MEAN = 0
STD = 1
# PATH = '../input/dogs-vs-cats-redux-kernels-edition/'
PATH = '../input/'
TRAIN_FOLDER = PATH+'train/'
TEST_FOLDER =  PATH+'test/'

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import os, cv2, random, glob
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from random import shuffle 
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from cv2 import imread, resize, cvtColor, imwrite
from keras.preprocessing.image import ImageDataGenerator
import gc
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam
#tf.keras.preprocessing.image
%matplotlib inline 
def read_image_and_resize(file_path, size=(128, 128), debug=False):
    img = imread(file_path, cv2.IMREAD_COLOR)
    img = cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = resize(img, size)
    if debug:
        import matplotlib.pyplot as plt
        print('Image resized from {} to {}'
              .format(img.shape, img_resized.shape))
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(img_resized)

    return img_resized


def load_image_dataset(
        dir_path='datasets/train/',
        size=(300, 300),
        isTrain=False):
    X, y = [], []
    
    if isTrain:
        all_img_files = glob.glob(os.path.join(dir_path, '*.jpg'))[:SAMPLE_SIZE]
    else:
        all_img_files = [os.path.join(dir_path, '%d.jpg' % i) for i in range(1, 12501)]
    for img_file in tqdm(all_img_files):
        img = read_image_and_resize(img_file, size=size)
        X.append(img)
        if isTrain:
            label = 'dog' in img_file and 1 or 0
            y.append(label)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y

def show_images_horizontally(images, labels=[], lookup_label=None,
                            figsize=(15, 7)):

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure, imshow, axis

    fig = figure(figsize=figsize)
    for i in range(images.shape[0]):
        fig.add_subplot(1, images.shape[0], i + 1)
        if lookup_label:
            plt.title(lookup_label[labels[i][0]])
        imshow(images[i], cmap='Greys_r')
        axis('off')
        
def plot_history(history, figsize=(15, 7)):
    # Plot the loss and accuracy curves for training and validation 
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure, axis

    fig = figure(figsize=figsize)
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

    
def gen_kaggle_sub(model, test):
    out_f = 'id,label'
    preds = model.predict(test)
    for idx, pred in enumerate(preds.reshape(-1)):
        out_f += f'\n{idx+1},{pred}'
    with open('submission.csv', 'w') as f:
        f.write(out_f)
    print('\nDone')
    
def img_transform(x):
    return (x - MEAN) / STD
    
def img_fit_transform(x):
    global MEAN, STD
    MEAN = x.mean()
    STD = x.std()
    print(f"MEAN:{MEAN}, STD:{STD}")
    return (x - MEAN) / STD
    
%%time
X, y = load_image_dataset(
        dir_path=TRAIN_FOLDER,
        size=(IMG_SIZE, IMG_SIZE),
        isTrain=True)

np.random.seed(RANDOM_STATE)
dataset_size = len(X)
perm = np.random.permutation(dataset_size)
X, y = X[perm], y[perm]  # X is only shuffled along it's first index.
print(X.shape, y.shape)
num_samples = 5
show_images_horizontally(X[:num_samples], y[:num_samples], figsize=(15, 10),
                         lookup_label={1: 'Dog', 0: 'Cat'})
train_set_ratio = 1 - TEST_SIZE
idx = int(dataset_size * train_set_ratio)
valid_set_size = X.shape[0] - idx
train_X, train_y, valid_X, valid_y = X[:idx], y[:idx], X[idx:], y[idx:]
print('Training set: {}, {}'.format(train_X.shape, train_y.shape))
print('Validation set: {}, {}'.format(valid_X.shape, valid_y.shape))
print('Some images in validation set:')
show_images_horizontally(valid_X[:num_samples], valid_y[:num_samples], figsize=(15, 10),
                         lookup_label={1: 'Dog', 0: 'Cat'})
# train_X = train_X/255
# valid_X = valid_X/255
train_X = img_fit_transform(train_X)
valid_X = img_transform(valid_X)
model = Sequential()
model.add(Conv2D(64,(3,3), input_shape=(IMG_SIZE,IMG_SIZE,3), activation="relu", padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(64,(3,3), activation="relu", padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation="relu", padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(128,(3,3),activation="relu", padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.33))

model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),activation="relu", padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(256,(3,3),activation="relu", padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(256,(3,3),activation="relu", padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.33))

model.add(Flatten())
model.add(BatchNormalization())
# model.add(Dense(1024,activation="relu", kernel_initializer='he_normal'))
model.add(Dense(256,activation="relu", kernel_initializer='he_normal'))
model.add(Dropout(0.5))    
model.add(Dense(1,activation="sigmoid"))

model.compile(
    loss="binary_crossentropy", 
#     optimizer=Adam(lr=0.001),
    optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
    metrics=["accuracy"]
)
model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.00001)
earlystop = EarlyStopping(monitor='val_acc', patience=15, verbose=2, restore_best_weights=True)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# history = model.fit(
#     x=train_X,
#     y=train_y,
#     validation_data=(valid_X, valid_y),
# #     validation_split=0.05,
#     epochs=25,
#     batch_size=BATCH_SIZE,
#     callbacks=[earlystop]
# )

history1 = model.fit_generator(
    datagen.flow(train_X, train_y, batch_size=BATCH_SIZE),
    steps_per_epoch=len(train_X) / BATCH_SIZE, 
    epochs=EPOCHS,
    validation_data=(valid_X, valid_y),
    callbacks=[earlystop, reduce_lr])
plot_history(history1)
del train_X, train_y, valid_X, valid_y, datagen
test, _ = load_image_dataset(TEST_FOLDER, size=(IMG_SIZE, IMG_SIZE), isTrain=False)
gen_kaggle_sub(model, img_transform(test))