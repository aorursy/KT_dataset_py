import pandas as pd

import numpy as np

import cv2

import sklearn

import sklearn.model_selection

import tensorflow.keras as keras



%matplotlib inline
# Download data & unzip if it doesn't already exist

import os.path

from io import BytesIO

from urllib.request import urlopen

from zipfile import ZipFile
def load_ext_file(data_zip_url, data_path='data/'):

    '''Download the zip file from URL and extract it to path (if specified).

    '''

    # Check if path already exits

    if not os.path.exists(data_path):

        with urlopen(data_zip_url) as zip_resp:

            with ZipFile(BytesIO(zip_resp.read())) as zfile:

                # Extract files into the data directory

                zfile.extractall(path=None)

       
# Use particular release for data from a simulation run

load_ext_file('https://github.com/MrGeislinger/clone-driving-behavior/releases/download/v0.14.0/data.zip')
def create_img_meas_dfs(log_csv, data_dir=None, orig_dir=None, skiprows=None):

    '''Creates DataFrames for the image paths and measurements using CSV path.

    

    Returns tuple of two DataFrames.

    '''

    data_header = [

        'image_center',

        'image_left',

        'image_right',

        'steer_angle', # [-1,1]

        'throttle', # boolen (if accelerating)

        'break', # boolean (if breaking)

        'speed' # mph

    ]



    df = pd.read_csv(

        log_csv,

        names=data_header,

        skiprows=skiprows

    )



    # Replace the original directory from dataset (if specified)

    if orig_dir and data_dir:

        for col in ['image_center','image_left','image_right']:

            df[col] = df[col].str.replace(orig_dir,data_dir)

    

    # Get specifics for each DF

    df_img_paths = df.iloc[:,:3]

    df_measurments = df.iloc[:,3:]

    

    return df_img_paths,df_measurments, df
df_imgs, df_meas, df_all = create_img_meas_dfs(

    log_csv='data/driving_log.csv', 

    skiprows=1)



display(df_imgs.head())



print('Stats for measurements:')

display(df_meas.describe())
ax = df_meas.steer_angle.hist(bins=50)
def flip_image(image, target):

    '''Horizontally flip image and target value.

    '''

    image_flipped = np.fliplr(image)

    target_flipped = -target

    return image_flipped, target_flipped

    
def adjust_offcenter_image(image, target, correction: float = 1e-2,

                                  img_camera_type: str = 'center'):

    '''

    img_camera_type: The type of camera image

    target: The target value (to be adjusted)

    '''

    # TODO: Adjust the target slightly for off-center image

    if img_camera_type == 'left':

        new_target = target + correction

    elif img_camera_type == 'right':

        new_target = target - correction

    # Don't make any correction if unknown or centere

    else: 

        new_target = target

    return image, new_target
def skip_low_steering(steer_value, steer_threshold=0.05, drop_percent=0.2):

    '''

    '''

    # Keep value if greater than threshold or by chance

    return (steer_value < steer_threshold['left']

            or steer_value > steer_threshold['right']

            or np.random.rand() > drop_percent)

        
def translate_image (image, target, correction=100, scale_factor=0.2):

    '''

    '''

    # Translate the image randomly about correction factor (then scaled)

    adjustment = int(correction * np.random.uniform(-1*scale_factor, scale_factor))

    # Get a new

    target_new = target + (adjustment / correction)

    n,m,c=image.shape

    bigsquare=np.zeros((n,m+100,c),image.dtype) 

    if adjustment < 0:

        bigsquare[:,:m+adjustment]=image[:,-adjustment:m]

    else:

        bigsquare[:,adjustment:m+adjustment]=image

    return bigsquare[:n,:m,:], target_new

def data_generator(X, y ,batch_size=64, center_only=False, data_dir='data/'):

    '''

    Generate a batch of training images and targets from a DataFrame.

    

    Inputs:

        X: array-like of paths to images

        y: array-like of targerts (in order of X)

    '''

    # Loop forever so the generator never terminates

    while True:

        # Shuffle the image paths and targets

        X_final = []

        y_final = []

        X_shuffled, y_shuffled = sklearn.utils.shuffle(X, y, n_samples=batch_size)

        # We grab the first element since there is 1 column

        for img_path,target in zip(X_shuffled,y_shuffled):

            fname = data_dir+img_path

            img = cv2.imread(fname[0])

            # Skip specifically for the center image (still checks left/right)

            steer_thresh = {'left':-0.01, 'right':0.005}

            drop_ratio = 0.3

            

            if skip_low_steering(target[0], steer_thresh, drop_ratio):

                X_final.append(img)

                y_final.append(target[0])

                # Use horizontally flipped images (new target)

                img_flipped, target_flipped = flip_image(img,target)

                X_final.append(img_flipped)

                y_final.append(target_flipped[0])

            # Check if we should use all images or just center

            if not center_only:

                # Translate the image randomly

                img_trans, target_trans = translate_image(img, target[0], scale_factor=0.5)

                X_final.append(img_trans)

                y_final.append(target_trans)

                

                # Order: center, left, right

                # Corret left image target & add image with target to array

                img_l = cv2.imread(fname[1])

                img_l, target_l = adjust_offcenter_image(img_l, target, 0.25, 'left')

                # Corret right image target & add image with target to array                

                img_r = cv2.imread(fname[2])

                img_r, target_r = adjust_offcenter_image(img_r, target, 0.25, 'right')



                X_final.append(img_l)

                y_final.append(target_l[0])

                # Use horizontally flipped images (new target)

                img_flipped, target_flipped = flip_image(img_l,target_l)

                X_final.append(img_flipped)

                y_final.append(target_flipped[0])



                X_final.append(img_r)

                y_final.append(target_r[0])

                # Use horizontally flipped images (new target)

                img_flipped, target_flipped = flip_image(img_r,target_r)

                X_final.append(img_flipped)

                y_final.append(target_flipped[0])



                



              

        batch_x = np.array(X_final)

        batch_y = np.array(y_final)

        # Ensures that targets remain as float32

        yield (batch_x, batch_y.astype('float32'))
# Note the multiple images will keep the proper shape

temp_generator = data_generator(

    df_all[['image_center','image_left','image_right']].values,

    df_all['steer_angle'].values.reshape(-1,1)

)
imgs,targets = next(temp_generator)
# Test to see if image reading works

import matplotlib.pyplot as plt



fig, axes = plt.subplots(1,4, figsize=(20, 20))

fig.subplots_adjust(wspace=.2)

axes = axes.ravel()



for i in range(4):

    axes[i].imshow(imgs[i])
# Adjust the target for off-center images to be used in training

X = df_all[['image_center','image_left','image_right']].values

y = df_all[['steer_angle']].values
X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(

                                        X, y, test_size=0.2, random_state=27)
# Using reasobable batch size so GPU can process enough images

train_generator = data_generator(X_train, y_train, batch_size=64)

valid_generator = data_generator(X_valid, y_valid, batch_size=64)
# Creating a resuable default convolution

from functools import partial

DefaultConv2D = partial(keras.layers.Conv2D, kernel_initializer='he_normal',

                        kernel_size=3, activation='elu', padding='SAME')
input_shape = (160,320,3)
# Based on https://developer.nvidia.com/blog/deep-learning-self-driving-cars/

model_list = [

    # Normalize the images

    keras.layers.Lambda(lambda x: (x/255.0) - 0.5, input_shape=input_shape),

    DefaultConv2D(filters=24, kernel_size=5),

    keras.layers.MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=36, kernel_size=5),

    keras.layers.MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=36, kernel_size=5),

    keras.layers.MaxPooling2D(pool_size=2),     

    keras.layers.Dropout(0.4),  # Dropout to regularize

    DefaultConv2D(filters=48),

    keras.layers.MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=64),

    keras.layers.MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=64),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Dropout(0.4),  # Dropout to regularize

    # Fully connected network

    keras.layers.Flatten(),

    keras.layers.Dense(units=1024, activation='relu'),

    keras.layers.Dropout(0.2),  # Dropout to regularize

    keras.layers.Dense(units=128, activation='relu'),

    keras.layers.Dropout(0.2),  # Dropout to regularize

    keras.layers.Dense(units=64, activation='relu'),

    keras.layers.Dense(units=16, activation='relu'),

    keras.layers.Dense(units=1)

]
# Adding in model to crop images first

model_list = (

    [model_list[0]] +

    # Crop out "unnecessary parts of the image"

    [keras.layers.Cropping2D(cropping=((60,20), (0,0)))] +

    model_list[1:]

)
model = keras.models.Sequential(model_list)
model.compile(

    loss='mse', 

    optimizer='nadam'

)
model.summary()
# Allow early stopping after not changing

stop_after_no_change = keras.callbacks.EarlyStopping(

    monitor='val_loss',

    patience=15,

    restore_best_weights=True

)
history = model.fit(

    x=train_generator,

    y=None, # Since using a generator

    batch_size=None, # Since using a generator

    epochs=256, # Large since we want to ensure we stop by early stopping

    steps_per_epoch=64, # Ideal: steps*batch_size = # of images

    validation_data=valid_generator,

    validation_steps=32,

    callbacks=[stop_after_no_change]

)
import matplotlib.pyplot as plt

%matplotlib inline 



def eval_model(model, model_history, X, y, show=True):

    '''

    '''

    score = model.evaluate(X, y)

    print(f'Loss: {score:.2f}')



    if show:

        plt.plot(model_history.history['loss'], label='Loss (training data)')

        plt.plot(model_history.history['val_loss'], label='Loss (validation data)')

        plt.ylabel('Loss')

        plt.xlabel('Epoch')

        plt.legend(loc='upper right')

        plt.show()
test_generator = data_generator(X_valid, y_valid, batch_size=64)

X_test, y_test = next(test_generator)

eval_model(model, history, X_test, y_test)
# Ignore the first epoch since it's typically very high compared to the rest

plt.plot(history.history['loss'], label='Loss (training data)')

plt.plot(history.history['val_loss'], label='Loss (validation data)')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.ylim(

    top=np.median(history.history['val_loss'])+np.std(history.history['val_loss']), 

    bottom=0.0

)

plt.legend(loc='upper right')

plt.show()
model.save('model.h5')
# Clean up the data downloaded (not needed for output)

!rm -rf data/
!rm -rf __MACOSX/