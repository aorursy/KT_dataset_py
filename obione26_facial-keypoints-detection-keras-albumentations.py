import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import albumentations as A # Should be version 0.4.6

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Input, Dense, Flatten

from tensorflow.keras.layers import Convolution2D, MaxPool2D, Dropout, BatchNormalization, Dropout, LeakyReLU

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import ResNet50

from tensorflow.python.keras.utils.data_utils import Sequence
# Reproducible results (set to None if you want to keep the randomness)

import os

import random



SEED = 1

np.random.seed(SEED)

tf.random.set_seed(SEED)

random.seed(SEED)



if SEED is not None:

    os.environ['PYTHONHASHSEED'] = str(SEED)
!mkdir ../data/

!unzip -q /kaggle/input/facial-keypoints-detection/training.zip -d ../data/train

!unzip -q /kaggle/input/facial-keypoints-detection/test.zip -d ../data/test
# Helper functions of this section

def format_dataset(dataframe):

    X = np.array([ format_image(x) for x in dataframe['Image'] ])

    

    if len(dataframe.columns) > 2:

        y = dataframe.drop('Image', axis=1).values

        return X, y

    

    return X



def format_image(img_row):

    """ Extract image from a pandas DataFrame row and normalize it in a [0,1] range. """

    img = img_row.split(' ')

    img = np.array(img, dtype=np.float32)

    img = img.reshape((96,96,1))

    img = img / 255.

    return img



def format_keypoints(keypoint):

    """ Normalize keypoints coordinates to lie in a [-1,1] range. """

    return (keypoint - 48.) /  48.



def unformat_keypoints(keypoint):

    """ Unormalize keypoints coordinates to lie in a [0,96] range. """

    return keypoint*48 + 48



def show_sample(img, keypoints, axis=None, color='b'):

    """ Display the target keypoints on top of the input image. """

    if axis is None:

        fig, axis = plt.subplots()

    

    axis.scatter(keypoints[0::2], keypoints[1::2], s=10, c=color)

    axis.imshow(img.squeeze(), cmap='gray')



def show_random_samples(X, y, n_rows=2, n_cols=5):

    """ Display a random subset of image-keypoints samples. """

    fig = plt.figure(figsize=(2*n_cols, 2*n_rows), dpi=100)



    for i, idx in enumerate(np.random.randint(0, len(y), n_rows*n_cols)):

        axis = fig.add_subplot(n_rows, n_cols, i+1, xticks=[], yticks=[])

        show_sample(X[idx], y[idx], axis=axis)

        axis.set_title(f'Sample #{idx}')
# Read data

train_dir = '../data/train/training.csv'

test_dir = '../data/test/test.csv'



train_data = pd.read_csv(train_dir)

test_data = pd.read_csv(test_dir)
# Data overview

train_data.sample(5).T
# Check the target keypoints statistics

train_data.describe()
# Check for missing training data

print(f'Train sample: {len(train_data)}')



print('Pourcentage of missing values:')

train_data.isna().mean().round(4) * 100
# Impute missing values



# Solution 1 - Drop all samples with one or multiple missing values

# train_data.dropna(inplace=True)



# Solution 2 - Replace NaN values by last valid value

# train_data.fillna(method = 'ffill', inplace=True)



# Solution 3 - Replace NaN values with each feature median

train_data.fillna(train_data.describe().T['50%'], inplace=True)



# Solution 4 - Replace NaN values with each feature mean

# train_data.fillna(train_data.describe().T['mean'], inplace=True)



# Solution 5 - DON'T RUN THIS CELL (go directly to section 4)



# Check imputed data

train_data.sample(5).T
# Format the data

X_train, y_train = format_dataset(train_data)

X_test = format_dataset(test_data)

print(X_train.shape, y_train.shape, X_test.shape)
# Display a random subset of training samples

show_random_samples(X_train, y_train)
# Helper functions of this section

def show_random_preds(model, X, n_rows=2, n_cols=5):

    fig = plt.figure(figsize=(2*n_cols, 2*n_rows), dpi=100)



    for i, idx in enumerate(np.random.randint(0, len(X), n_rows*n_cols)):

        X_input = X[idx:idx+1, ...]

        y_pred = model.predict(X_input).flatten()

        

        axis = fig.add_subplot(n_rows, n_cols, i+1, xticks=[], yticks=[])

        show_sample(X_input.squeeze(), y_pred, axis=axis)

        axis.set_title(f'Sample #{idx}')



def plot_loss(hist, metric='loss'):

    plt.plot(hist.history[metric])

    plt.plot(hist.history[f'val_{metric}'])

    plt.title(f'{metric.upper()} vs Epoch')

    plt.ylabel(metric.upper())

    plt.xlabel('Epochs')

    plt.legend(['train', 'validation'], loc='upper right')

    plt.show()
print(f'Input size: {X_train.shape}')

print(f'Output size: {y_train.shape}')
# Architecture 1

def create_small_dense_network():

    model = Sequential()

    model.add(Input(shape=(96, 96, 1)))

    model.add(Flatten())

    model.add(Dense(units=100, activation='relu'))

    model.add(Dense(units=30, activation=None))



    return model
# Train model

model1 = create_small_dense_network()



es = EarlyStopping(monitor='val_loss', patience=10)

mc = ModelCheckpoint('best_model1.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)



model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

hist1 = model1.fit(X_train, y_train, epochs=500, batch_size=256, verbose=0, validation_split=0.2, callbacks=[es, mc])
# Show training and validation loss

plot_loss(hist1, metric='mae')
# Visualize prediction on random test samples

model1.load_weights('best_model1.h5')

show_random_preds(model1, X_test)
# Architecture 2

def create_convnet(n_outputs=30):

    model = Sequential()



    model.add(Convolution2D(32, (5,5), padding='same', input_shape=(96,96,1)))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(Convolution2D(32, (5,5), padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))



    model.add(Convolution2D(64, (3,3), padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(Convolution2D(64, (3,3), padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))



    model.add(Convolution2D(96, (3,3), padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(Convolution2D(96, (3,3), padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))



    model.add(Convolution2D(128, (3,3),padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(Convolution2D(128, (3,3),padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))



    model.add(Convolution2D(256, (3,3),padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(Convolution2D(256, (3,3),padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))



    model.add(Convolution2D(512, (3,3), padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(Convolution2D(512, (3,3), padding='same'))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(n_outputs))



    return model
# Train model

model2 = create_convnet()



es = EarlyStopping(monitor='val_loss', patience=10)

mc = ModelCheckpoint('best_model2.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)



model2.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

hist2 = model2.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0, validation_split=0.10, callbacks=[es, mc])
# Show training and validation loss

plot_loss(hist2, metric='mae') # 1.377 -> 1.298
# Visualize prediction on random test samples

model2.load_weights('best_model2.h5')

show_random_preds(model2, X_test)
# Helper functions of this section

class DataLoader(Sequence):

    def __init__(self, X, y, batch_size, augmentations=None, as_rgb=False):

        self.X, self.y = X, y

        self.batch_size = batch_size

        self.augment = augmentations

        self.shuffle = True

        self.as_rgb = as_rgb

        self.on_epoch_end()



    def __len__(self):

        """ Corresponds to the number of steps in one epoch. """

        return int(np.ceil(len(self.X) / float(self.batch_size)))



    def __getitem__(self, idx):

        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]

        batch_X = self.X[indexes, ...]

        batch_y = self.y[indexes, :]

        

        # Convert grayscale to RGB if needed (if you want to use a pre-trained ResNet for example)

        if self.as_rgb:

            batch_X = np.tile(batch_X, reps=(1,1,1,3))



        # Apply transformations on both images and keypoints

        if self.augment is not None:

            keypoints = np.array([ tuple(zip(point[::2], point[1::2])) for point in batch_y ])

            transformed = [ self.augment(image=x, keypoints=y) for x,y in zip(batch_X, keypoints) ]

            batch_X = np.stack([ z['image'] for z in transformed ], axis=0)

            batch_y = np.stack([ np.array(z['keypoints']).flatten(order='C') for z in transformed ], axis=0)



        return batch_X, batch_y



    def on_epoch_end(self):

        """ Shuffle the data after each epoch to avoid oscillation patterns in the loss. """

        self.indexes = np.arange(len(self.X))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)
# Add splitting to avoid leakage

X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, test_size=0.10, shuffle=True)



# Define augmentation strategy

transform = A.Compose([

    A.ShiftScaleRotate(rotate_limit=30, p=0.5),

    A.RandomBrightnessContrast(p=0.5),

    A.GaussianBlur(p=0.5),

    A.GaussNoise(var_limit=(1e-5, 1e-3), p=0.5),

], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))



train_loader = DataLoader(X_train2, y_train2, batch_size=128, augmentations=transform)

print(X_train2.shape, y_train2.shape)

print(X_valid.shape, y_valid.shape)



# Visualize augmented data

x_batch, y_batch = train_loader[1]

show_random_samples(x_batch.squeeze(), y_batch)
es = EarlyStopping(monitor='val_loss', patience=20)

mc = ModelCheckpoint('best_model3.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)



model3 = create_convnet()

model3.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

hist3 = model3.fit(train_loader, steps_per_epoch=len(train_loader),

                   validation_data=(X_valid, y_valid),

                   epochs=500, verbose=0, callbacks=[es, mc])
plot_loss(hist3, metric='mae')
model3.load_weights('best_model3.h5')

show_random_preds(model3, X_test)
def create_specialist(n_outputs=30, weights=None, freeze=False):

    model = create_convnet()

    

    if weights is not None:

        model.load_weights(weights)

    

    if freeze:

        for layers in model.layers[:-10]:

            layers.trainable = False

        

    if n_outputs != 30:

        model.layers.pop()

        model.add(Dense(n_outputs))

    return model



def train_specialist(model, keypoints_range, model_name):

    # Prepare dataset

    train_data = pd.read_csv(train_dir)

    

    select_col_idx = list(range(*keypoints_range[model_name])) + [-1]

    subdata = train_data.iloc[:, select_col_idx]

    subdata = subdata.dropna()

    

    X_train, y_train = format_dataset(subdata)

    X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, test_size=0.10, shuffle=True)



    # Define augmentation strategy

    transform = A.Compose([A.ShiftScaleRotate(rotate_limit=30, p=0.5),

                           A.RandomBrightnessContrast(p=0.5),

                           A.GaussianBlur(p=0.5),

                           A.GaussNoise(var_limit=(1e-5, 1e-3), p=0.5)],

                          keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))



    train_loader = DataLoader(X_train2, y_train2, batch_size=128, augmentations=transform)

    

    # Train specialist model

    es = EarlyStopping(monitor='val_loss', patience=10)

    mc = ModelCheckpoint(f'best_model_{model_name}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)



    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

    hist = model.fit(train_loader, steps_per_epoch=len(train_loader),

                     validation_data=(X_valid, y_valid),

                     epochs=250, verbose=0, callbacks=[es, mc])

    model.load_weights(f'best_model_{model_name}.h5')    

    

    return model, hist





class ConcatenateSpecialists:

    def __init__(self, models):

        self.models = models

        

    def predict(self, X):

        return np.hstack([ m.predict(X) for m in self.models ])

    

# TODO: https://www.deeplearningbook.org/contents/regularization.html (page 245-246) -> retrain on whole dataset  
specialist_keypoints = {'eyes_centers':(0,4), 'eyes_corners':(4,12), 'eyebrows': (12,20), 'nose': (20,22), 'mouth': (22,30)}

models = {}



for region, keypoint_ids in specialist_keypoints.items():

    print(f'Training model {region}...')

    model = create_specialist(n_outputs=keypoint_ids[1]-keypoint_ids[0], weights='best_model3.h5', freeze=False)

    models[region] = train_specialist(model, specialist_keypoints, region)
model4 = ConcatenateSpecialists([m[0] for m in models.values()])

show_random_preds(model4, X_test)
model = model3

layer_names = [layer.name for layer in model.layers]

intermediate_outputs = [layer.output for layer in model.layers]

visualization_model = Model(inputs = model.input, outputs = intermediate_outputs)



n = 10 # number of random filters to display per layer

x = X_test[0:2]

feature_maps = visualization_model.predict(x)



for layer_name, feature_map in zip(layer_names, feature_maps):    

    if len(feature_map.shape) == 4:

        # Just do this for the conv / maxpool layers, not the fully-connected layers

        n_features = feature_map.shape[-1] 

        size = feature_map.shape[ 1]



        # We will tile our images in this matrix

        display_grid = np.zeros((size, size * n))



        # Postprocess the feature to be visually palatable

        for i, idx in enumerate(np.random.randint(0, n_features, n)):

            x  = feature_map[0, :, :, i]

            x -= x.mean()

            x /= x.std ()

            x *=  64

            x += 128

            x  = np.clip(x, 0, 255).astype('uint8')

            display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid



        # Display the grid

        scale = 3

        plt.figure(figsize=(scale * n, scale))

        plt.title(layer_name)

        plt.grid(False)

        plt.imshow(display_grid, aspect='auto', cmap='viridis')
def create_submission_file(model, X_test, save_name='model_preds'):

    predictions = model.predict(X_test)

    print(f'Shape: {predictions.shape} - Min: {predictions.min()} - Max: {predictions.max()}')



    # Post-process predictions

    predictions[predictions > 96] = 96

    

    # Lookup table filters out the expected prediction points landmarks for each test image

    lookid_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')



    image_id = list(lookid_data['ImageId']-1)

    landmark_names = list(lookid_data['FeatureName'])

    landmark_ids = [ landmark_names.index(f) for f in landmark_names ]



    expected_preds = [ predictions[x,y] for x,y in zip(image_id, landmark_ids) ]



    rowid = pd.Series(lookid_data['RowId'], name = 'RowId')

    loc = pd.Series(expected_preds, name = 'Location')

    submission = pd.concat([rowid, loc], axis = 1)

    

    submission.to_csv(f'{save_name}.csv',index = False)

    print(f'Successfully created {save_name}.csv !')
create_submission_file(model1, X_test, 'model_preds1')

create_submission_file(model2, X_test, 'model_preds2')

create_submission_file(model3, X_test, 'model_preds3')

create_submission_file(model4, X_test, 'model_preds4')