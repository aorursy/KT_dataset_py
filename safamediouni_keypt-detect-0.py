import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import skimage.transform as sk

from sklearn.model_selection import train_test_split

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, load_model 

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, SeparableConv2D, MaxPool2D

from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

from keras.optimizers import Adam



%matplotlib inline
# set up path to files

dir_path = os.path.join('../input', os.listdir('../input')[0])

print("Files are :", os.listdir(dir_path))
def load_data(dir_path, filename):#, cols=None):



    df = pd.read_csv(os.path.join(dir_path, filename))



    # The Image column has pixel values saved as strings separated by a space

    if filename not in ['IdLookupTable.csv','SampleSubmission.csv']:

        df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' '))



    return df
def summary(df, test=False):

    

    dataname = 'test' if test else 'train'  

    

    print("Length of %s data %d..\n" % (dataname, len(df)))

    print("Count values for each variable:\n ")

    print(df.count(), '\n')

    print("How many variables has missing values ?\n ")

    print( df.isnull().any().value_counts())

    print()

    print("Pourcentage of missing values for each variable: \n")

    summary_list =[100 - df[c].count()/len(df)*100 for c in df.columns]

    var_list = [var for var in df.columns]



    for i in range(len(summary_list)):

        print("{} : {}".format(var_list[i], np.round(summary_list[i]),2))

        print()
def get_mean_cols(df):

    """

    return mean over means of variables for a df

    """

    df_mean_cols = df[df.columns[:-1]].mean(axis = 0, skipna = True).reset_index()

    mean_cols = df_mean_cols[0].mean(axis = 0, skipna = True)

    

    return round(mean_cols)
def scale_data(df, test=False, random_state=42):

    """

    Scale pixel values to [0,1] for train and test set

    Scale keypoints to [-1, 1], only for train set, as there is no target values for test set

    test: True, if test, otherwise train

    """

    img = np.vstack(df['Image'].values) / 255. # scale Image range from [0, 255] to [0, 1]

    img = img.astype(np.float32) # change data type to float

    mean_cols = get_mean_cols(df) # get mean over variables

    

    if not test:

        keypoints = df[df.columns[:-1]].values # scale keypoints to be centered around 0 with a range of [-1, 1]

        keypoints = (keypoints - mean_cols) / mean_cols  # scale keypoints to [-1, 1]

        keypoints = keypoints.astype(np.float32) # change keypoints type to float

    else:

        keypoints = None

        

    return img, keypoints, mean_cols
def reshape_data(df,width=96, height=96):

    """

    Reshape data to (len_df, 96, 96, ?) by default

    """

    return df.reshape(df.shape[0], width, height, -1) 
def _random_indices(inputs, ratio, random_state=1234):

    """Generate random unique indices according to ratio"""

    np.random.seed(random_state);

    actual_batchsize = inputs.shape[0]

    size = int(actual_batchsize * ratio)

    indices = np.random.choice(actual_batchsize, size, replace=False)

    return indices
def rotate(y, inputs, targets, rotate_ratio, angle= None, right_left = 0):

    """Rotate slighly the image and the targets. Works only with one channel"""

    if angle is None:

        angle = np.random.randint(10)

    if right_left != 0:

        angle =  360 - angle

    for i in range(inputs.shape[0]):

        inputs[i, :, :, 0] = sk.rotate(inputs[i, :, :, 0], angle)

    angle = np.radians(angle)

    indices = np.arange(targets.shape[0])

    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    targets = targets.reshape(len(targets), y.shape[1] // 2, 2)

    targets[indices] = np.dot(targets[indices], R)

    targets = targets.reshape(len(targets), y.shape[1])

    

    return inputs, targets
def flipp(inputs, targets, flip_ratio, flip_indices= None, random_seed=123):

    """Flip image"""

    if flip_indices is None:

        flip_indices = [ (0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11),

                        (12, 16), (13, 17), (14, 18), (15, 19), (22, 24),

                        (23, 25) ]

    for i in range(inputs.shape[0]):

        inputs[i, :, :, :] = inputs[i, :, ::-1, :]

    indices = np.arange(inputs.shape[0])

    targets[indices, ::2] = targets[indices, ::2] * -1

    for a, b in flip_indices:

        targets[indices, a], targets[indices, b] = targets[indices, b], targets[indices, a]

    return inputs, targets
# idea from : https://www.kaggle.com/balraj98/data-augmentation-for-facial-keypoint-detection

def brightness(inputs):

    """ Contrast jittering (reduction)"""

    in_brightness_images = np.clip(inputs*1.6, 0.0, 1.0)

    

    return in_brightness_images
def random_shift(shift_range,n=96):

    '''

    shift_range: 

    The maximum number of columns/rows to shift

    return: 

    keep(0):   minimum row/column index to keep

    keep(1):   maximum row/column index to keep

    assign(0): minimum row/column index to assign

    assign(1): maximum row/column index to assign

    shift:     amount to shift the landmark



    assign(1) - assign(0) == keep(1) - keep(0)

    '''

    shift = np.random.randint(-shift_range, shift_range)

    def shift_left(n,shift):

        shift = np.abs(shift)

        return(0,n - shift)

    def shift_right(n,shift):

        shift = np.abs(shift)

        return(shift,n)



    if shift < 0:

        keep = shift_left(n,shift) 

        assign = shift_right(n,shift)

    else:

        assign = shift_left(n,shift) ## less than 96

        keep = shift_right(n,shift)



    return((keep,  assign, shift))
def shift_single_image(x_,y_,w=96, h=96, prop=0.1):

    '''

    x_: a single picture array (96, 96, 1)

    y_: 15 landmark locations 

               [0::2] contains x axis values

               [1::2] contains y axis values 

    prop: proportion of random horizontal and vertical shift

          relative to the number of columns

    '''

    w_shift_max = int(w * prop)

    h_shift_max = int(h * prop)



    w_keep,w_assign,w_shift = random_shift(w_shift_max)

    h_keep,h_assign,h_shift = random_shift(h_shift_max)

    

    x_new = np.ones(x_.shape)

    y_new = np.ones(y_.shape)

    

    x_new[w_assign[0]:w_assign[1], h_assign[0]:h_assign[1]] = x_[w_keep[0]:w_keep[1], h_keep[0]:h_keep[1]]



    y_new[0::2] = y_[0::2] - h_shift/float(w/2.)

    y_new[1::2] = y_[1::2] - w_shift/float(h/2.)

    return(x_new,y_new)
def shift_image(X,y,prop=0.1):

    X = X.reshape(-1,96,96)

    y = y.reshape(-1,30)

    for i in range(X.shape[0]):

        x_ = X[i]

        y_ = y[i]

        X[i],y[i] = shift_single_image(x_,y_,prop=prop)

    return(X,y)
def add_noise(inputs, noise_ratio=0.001):

    noisy_img = np.zeros(inputs.shape)

    for i in range(inputs.shape[0]):

        noise = np.random.randn(96,96,1)

        noisy_img[i] = inputs[i] + noise_ratio*noise

    return noisy_img
def plot_img(df, keypoints, mean_cols, title="", add_keypoints=False ,num_img=None, rand_img=False):

    """

    Plots different images randomly selected from df

    num_img : numbre of images to select

    add_keypoints = True if you want to add keypoints to each image

    """

    if num_img is None:

        num_img = df.shape[0]

        

    if rand_img:

        list_img = list(np.random.choice(np.arange(0,df.shape[0]), num_img, replace=False)) # select random num_img index

    else:

        list_img = np.arange(num_img)

    

    fig = plt.figure(figsize=(12, 12))

    fig.suptitle(title, fontsize='x-large')

    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, hspace=0.05, wspace=0.05)  

    j = 0

    for i in list_img: # i index for image selected

        axis = fig.add_subplot(4, 4, j + 1, xticks=[], yticks=[])

        img = plt.imshow(df[i].reshape(96,96), cmap='gray')

        # as we scaled keypoints to [-1,1], we have to retrieve the real values by *mean_cols*

        if add_keypoints:

            axis.scatter(keypoints[i][0::2]*mean_cols + mean_cols, keypoints[i][1::2]*mean_cols + mean_cols, marker='x', s=10)

        j += 1

    plt.show()
def split_train_validation(X, y, size=0.3, random_state = 69):

    

    X_train, X_validation, y_train, y_validation = train_test_split(X, y,  test_size=size, random_state=random_state)

    print("Splitting data into train {} and validation {}".format(X_train.shape, X_validation.shape))

    return X_train, X_validation, y_train, y_validation
def plot_loss(hist,name,plt):



    loss = hist['loss']

    val_loss = hist['val_loss']

        

    plt.plot(loss,"--", linewidth=3,label="Train:" + name)

    plt.plot(val_loss, linewidth=3,label="Validation:" + name)
train_data = load_data(dir_path, 'training.zip')

test_data = load_data(dir_path, 'test.zip')

idlookup = load_data(dir_path, 'IdLookupTable.csv')

sample_submission = load_data(dir_path,'SampleSubmission.csv')
train_data.head().T # transpose is a simpler way to see how our training set looks like
test_data.head()
sample_submission.head()
summary(train_data)
summary(test_data, test=True)
summary(idlookup)
# first of all we will create a copy of our current dataset so that nothing is lost

train_data_copy = train_data.copy()

test_data_copy = test_data.copy()
train_dropna = train_data_copy.dropna()

print("Tech: drop NaNs, X train shape: {}\n".format(train_dropna.shape))



# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html

train_fill_nan = train_data_copy.fillna(method = 'ffill')

print("Tech: fill NaNs with forward values, X train shape: {}\n".format(train_fill_nan.shape))
# check if there still missing values in each of our datasets

print("Missing values for the train set where simply NaNs were dropped:\n")

print(train_dropna.isnull().any().value_counts())

print()

print("Missing values for the train set where NaNs filled using the forward technique:\n")

print(train_fill_nan.isnull().any().value_counts())

X_train_dropna, y_train_dropna, mean_cols = scale_data(train_dropna) # x_train == image, y_train == keypoints

X_test_dropna, y_test_dropna, _ = scale_data(train_dropna, test=True)

X_train_fill_nan, y_train_fill_nan, mean_cols = scale_data(train_fill_nan)

X_test_fill_nan, y_test_fill_nan, _ = scale_data(train_fill_nan, test=True)
print("1. Drop missing values:")

print()

print("X train shape: {}, y train shape: {}, X test shape: {}".format(X_train_dropna.shape, y_train_dropna.shape, X_test_dropna.shape))

print()

print("2. Fill missing values:")

print()

print("X train shape: {}, y train shape: {}, X test shape: {}".format(X_train_fill_nan.shape, y_train_fill_nan.shape, X_test_fill_nan.shape))
# reshape X train and X test

X_train_dropna = reshape_data(X_train_dropna)

X_test_dropna = reshape_data(X_test_dropna)

X_train_fill_nan = reshape_data(X_train_fill_nan)

X_test_fill_nan = reshape_data(X_test_fill_nan)

print("X train shape: {}, X test shape: {}".format(X_train_dropna.shape, X_test_dropna.shape))

print("X train shape: {}, X test shape: {}".format(X_train_fill_nan.shape, X_test_fill_nan.shape))
title="Example of images in our train set "

plot_img(X_train_dropna, y_train_dropna, mean_cols, title, True, 4)
flip_ratio = 0.8

rotate_ratio = 0.8

contrast_ratio = 1.2

random_seed = 342

angle = 9

use_flip_transf = True

use_rotation_transf = False

use_brightness_transf = False
aug_x_train = X_train_dropna.copy()

aug_y_train = y_train_dropna.copy()
if use_flip_transf:

    flipped_img, flipped_kepoints = flipp(aug_x_train, aug_y_train, flip_ratio, None, random_seed)

    print("Shape of flipped images {} and keypoints {}".format(flipped_img.shape, flipped_kepoints.shape))
title = "Flipped images"

plot_img(df=flipped_img, keypoints=flipped_kepoints, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)

title = "Original images"

plot_img(df=X_train_dropna, keypoints=y_train_dropna, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)
aug_x_train = X_train_dropna.copy()

aug_y_train = y_train_dropna.copy()

use_rotation_transf = True

if use_rotation_transf:

    rotated_img_l, rotated_keypoints_l = rotate(y_train_dropna, aug_x_train, aug_y_train, rotate_ratio, 9, 0)
aug_x_train = X_train_dropna.copy()

aug_y_train = y_train_dropna.copy()

if use_rotation_transf:

    rotated_img_r, rotated_keypoints_r = rotate(y_train_dropna, aug_x_train, aug_y_train, rotate_ratio, 9, 1)

print("Shape of rotated images {} and keypoints {}".format(rotated_img_r.shape, rotated_keypoints_r.shape))
title = "Rotated images to the left"

plot_img(df=rotated_img_l, keypoints=rotated_keypoints_l, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)

title = "Rotated images to the right"

plot_img(df=rotated_img_r, keypoints=rotated_keypoints_r, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)

title = "Original images"

plot_img(df=X_train_dropna, keypoints=y_train_dropna, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)
aug_x_train = X_train_dropna.copy()

aug_y_train = y_train_dropna.copy()

use_brightness_transf = True

if use_brightness_transf:

    inc_brightness_images = brightness(aug_x_train)

    print("Shape of brightned images {} ".format(inc_brightness_images.shape))
title = "Increase brightness images"

plot_img(df=inc_brightness_images, keypoints=y_train_dropna, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)

title = "Original images"

plot_img(df=X_train_dropna, keypoints=y_train_dropna, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)
aug_x_train = X_train_dropna.copy()

aug_y_train = y_train_dropna.copy()
shifted_img, shifted_keypoints = shift_image(aug_x_train, aug_y_train, prop=0.1)

shifted_img = shifted_img[:,:,:,np.newaxis]

print("Shape of shifted images {} ".format(shifted_img.shape))
title = 'Shifted images'

plot_img(df=shifted_img, keypoints=shifted_keypoints, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)

title = 'Original images'

plot_img(df=X_train_dropna, keypoints=y_train_dropna, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)
aug_x_train = X_train_dropna.copy()

noisy_img = add_noise(aug_x_train)

print("Shape of noisy images {} ".format(noisy_img.shape))
title = 'Noisy images'

plot_img(df=noisy_img, keypoints=y_train_dropna, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)

title = 'Original images'

plot_img(df=X_train_dropna, keypoints=y_train_dropna, mean_cols=mean_cols, title=title, add_keypoints=True, num_img=4)
aug_x_train_ffill = X_train_fill_nan.copy().reshape((-1, 96,96,1))

aug_y_train_ffill = y_train_fill_nan.copy()

aug_x_train = X_train_dropna.copy().reshape((-1, 96,96,1))

aug_y_train = y_train_dropna.copy()

aug_x_train = np.concatenate((aug_x_train, flipped_img, rotated_img_r, rotated_img_l, inc_brightness_images, shifted_img, noisy_img))

aug_y_train = np.concatenate((aug_y_train, flipped_kepoints, rotated_keypoints_r, rotated_keypoints_l, aug_y_train, shifted_keypoints, aug_y_train))

print("Number of images in the new train dataset using data augmentation :{} {} ".format(aug_x_train.shape, aug_y_train.shape))
# drop NaN values

x_train_dna, x_validation_dna, y_train_dna, y_validation_dna = split_train_validation(X_train_dropna, y_train_dropna)

# impute NaN values

x_train_ffill, x_validation_ffill, y_train_ffill, y_validation_ffill = split_train_validation(X_train_fill_nan, y_train_fill_nan)

# Data augmentation

x_train_da, x_validation_da, y_train_da, y_validation_da = split_train_validation(aug_x_train, aug_y_train, 0.1)
# model_03_s = Sequential()



# model_03_s.add(Convolution2D(32, (3,3), input_shape=(96,96,1)))

# model_03_s.add(Activation('relu'))



# model_03_s.add(Convolution2D(64, (3,3)))

# model_03_s.add(Activation('relu'))

# model_03_s.add(MaxPool2D(pool_size=(2, 2)))

# model_03_s.add(Dropout(0.5))



# model_03_s.add(Convolution2D(128, (3,3)))

# model_03_s.add(Activation('relu'))



# model_03_s.add(Convolution2D(256, (3,3)))

# model_03_s.add(Activation('relu'))

# model_03_s.add(MaxPool2D(pool_size=(2, 2)))

# model_03_s.add(Dropout(0.5))



# model_03_s.add(Convolution2D(512, (3,3)))

# model_03_s.add(Activation('relu'))

# model_03_s.add(Dropout(0.5))



# model_03_s.add(Flatten())

# model_03_s.add(Dense(512,activation='relu'))



# model_03_s.add(Dense(30))

# model_03_s.summary()
# callbacks = [

#         EarlyStopping(monitor='val_loss', patience=15, mode='min',restore_best_weights=True, verbose=1),

#         ModelCheckpoint(filepath = 'best_model_03_s.hdf5', monitor='val_mae', verbose=1, save_best_only=True, mode='min')

#     ]



# hist_03_s = model_03_s.fit(x_train_da, y_train_da,

#             epochs= 100, batch_size=64,

#             validation_data=(x_validation_da, y_validation_da),

#             callbacks=callbacks,

#             verbose=1)
# plot_loss(hist_03_s.history,"Best model",plt)

# plt.legend()

# plt.grid()

# plt.yscale("log")

# plt.xlabel("epoch")

# plt.ylabel("log loss")

# plt.show()
model_06_01 = Sequential()



model_06_01.add(Convolution2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(96,96,1)))

model_06_01.add(Activation('relu'))

model_06_01.add(Dropout(0.1))





model_06_01.add(SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', use_bias=False))

model_06_01.add(Activation('relu'))

model_06_01.add(MaxPooling2D(pool_size=(2, 2)))

model_06_01.add(Dropout(0.1))





model_06_01.add(SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', use_bias=False))

model_06_01.add(Activation('relu'))

model_06_01.add(BatchNormalization())

model_06_01.add(MaxPooling2D(pool_size=(2, 2)))

model_06_01.add(Dropout(0.25))



model_06_01.add(SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', use_bias=False))

model_06_01.add(Activation('relu'))

model_06_01.add(MaxPooling2D(pool_size=(2, 2)))

model_06_01.add(Dropout(0.25))





model_06_01.add(Flatten())

model_06_01.add(Dense(512,activation='relu'))

model_06_01.add(Dropout(0.5))

model_06_01.add(Dense(30))

model_06_01.summary()

model_06_01.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics=['mae', 'acc'])
callbacks = [

        EarlyStopping(monitor='val_loss', patience=15, mode='min',restore_best_weights=True, verbose=1),

        ModelCheckpoint(filepath = 'best_model_06_01.hdf5', monitor='val_mae', verbose=1, save_best_only=True, mode='min')

    ]



hist_06_01 = model_06_01.fit(x_train_da, y_train_da,

            epochs= 80, batch_size=128,

            validation_data=(x_validation_da, y_validation_da),

            callbacks=callbacks,

            verbose=1)
plot_loss(hist_06_01.history,"Best model",plt)

plt.legend()

plt.grid()

plt.yscale("Log")

plt.xlabel("Epoch")

plt.ylabel("Log loss")

plt.show()
# model_06_02 = Sequential()



# model_06_02.add(Convolution2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(96,96,1)))

# model_06_02.add(LeakyReLU(alpha = 0.1))

# model_06_02.add(Dropout(0.1))





# model_06_02.add(SeparableConv2D(filters=256, kernel_size=(3,3), padding='same'))

# model_06_02.add(LeakyReLU(alpha = 0.1))

# model_06_02.add(MaxPooling2D(pool_size=(2, 2)))





# model_06_02.add(SeparableConv2D(filters=256, kernel_size=(3,3), padding='same'))

# model_06_02.add(LeakyReLU(alpha = 0.1))

# model_06_02.add(BatchNormalization())

# model_06_02.add(MaxPooling2D(pool_size=(2, 2)))

# model_06_02.add(Dropout(0.25))



# model_06_02.add(SeparableConv2D(filters=256, kernel_size=(3,3), padding='same'))

# model_06_02.add(LeakyReLU(alpha = 0.1))

# model_06_02.add(MaxPooling2D(pool_size=(2, 2)))

# model_06_02.add(Dropout(0.25))





# model_06_02.add(Flatten())

# model_06_02.add(Dense(512,activation='relu'))

# model_06_02.add(Dropout(0.5))

# model_06_02.add(Dense(30))

# model_06_02.summary()

# model_06_02.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics=['mae', 'acc'])
# callbacks = [

#         EarlyStopping(monitor='val_loss', patience=15, mode='min',restore_best_weights=True, verbose=1),

#         ModelCheckpoint(filepath = 'best_model_06_02.hdf5', monitor='val_mae', verbose=1, save_best_only=True, mode='min')

#     ]



# hist_06_02 = model_06_02.fit(x_train_da, y_train_da, batch_size=64, epochs=100,validation_data=(x_validation_da,y_validation_da),callbacks=callbacks, verbose=1)
# plot_loss(hist_06_02.history,"Best model",plt)

# plt.legend()

# plt.grid()

# plt.yscale("Log")

# plt.xlabel("Epoch")

# plt.ylabel("Log loss")

# plt.show()
# model_06_03 = Sequential()



# model_06_03.add(Convolution2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(96,96,1)))

# model_06_03.add(LeakyReLU(alpha = 0.1))

# # model_06_03.add(BatchNormalization())

# model_06_03.add(Dropout(0.1))





# model_06_03.add(SeparableConv2D(filters=256, kernel_size=(3,3), padding='same'))

# model_06_03.add(LeakyReLU(alpha = 0.1))

# # model_06_03.add(BatchNormalization())

# model_06_03.add(MaxPooling2D(pool_size=(2, 2)))

# model_06_03.add(Dropout(0.25))





# model_06_03.add(SeparableConv2D(filters=256, kernel_size=(3,3), padding='same'))

# model_06_03.add(LeakyReLU(alpha = 0.1))

# model_06_03.add(BatchNormalization())

# model_06_03.add(MaxPooling2D(pool_size=(2, 2)))

# model_06_03.add(Dropout(0.5))



# model_06_03.add(SeparableConv2D(filters=256, kernel_size=(3,3), padding='same'))

# model_06_03.add(LeakyReLU(alpha = 0.1))

# model_06_03.add(MaxPooling2D(pool_size=(2, 2)))

# model_06_03.add(Dropout(0.5))





# model_06_03.add(Flatten())

# model_06_03.add(Dense(512,activation='relu'))

# model_06_03.add(Dropout(0.5))

# model_06_03.add(Dense(30))

# model_06_03.summary()
# callbacks = [

#         EarlyStopping(monitor='val_loss', patience=15, mode='min',restore_best_weights=True, verbose=1),

#         ModelCheckpoint(filepath = 'best_model_06_03.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#     ]



# hist_06_03 = model_06_03.fit(x_train_da, y_train_da, batch_size=64, epochs=100,validation_data=(x_validation_da,y_validation_da),callbacks=callbacks, verbose=1)
# plot_loss(hist_06_03.history,"Best model",plt)

# plt.legend()

# plt.grid()

# plt.yscale("Log")

# plt.xlabel("Epoch")

# plt.ylabel("Log loss")

# plt.show()
test_,_, _ = scale_data(test_data_copy, True)

test_img = reshape_data(test_)
best_model = load_model('best_model_06_01.hdf5')

pred = best_model.predict(test_img)
feature_name = list(idlookup['FeatureName'])

image_id = list(idlookup['ImageId']-1)

row_id = list(idlookup['RowId'])



feature_list = []

for feature in feature_name:

    feature_list.append(feature_name.index(feature))

    

predictions = []

for x,y in zip(image_id, feature_list):

    predictions.append(pred[x][y])

    

row_id = pd.Series(row_id, name = 'RowId')

locations = pd.Series(predictions, name = 'Location')

locations = locations*mean_cols +mean_cols

submission_result = pd.concat([row_id,locations],axis = 1)

submission_result.to_csv('best_perf_15_1600.csv',index = False)