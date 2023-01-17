from glob import glob

import pandas as pd

import os

import json

import matplotlib.pyplot as plt

import h5py

import numpy as np

import seaborn as sns

sns.set_style("whitegrid", {'axes.grid' : False})

try:

    from tqdm import tqdm

except ImportError:

    print('Missing tqdm...')

    tqdm = lambda x: x

data_dir = os.path.join('..', 'input')
# load the data file and extract dimensions

with h5py.File(os.path.join(data_dir,'gaze.h5'),'r') as t_file:

    print(list(t_file.keys()))

    assert 'image' in t_file, "Images are missing"

    assert 'look_vec' in t_file, "Look vector is missing"

    look_vec = t_file['look_vec'].value

    assert 'path' in t_file, "Paths are missing"

    print('Images found:',len(t_file['image']))

    for _, (ikey, ival) in zip(range(1), t_file['image'].items()):

        print('image',ikey,'shape:',ival.shape)

        img_width, img_height = ival.shape

    syn_image_stack = np.stack([a for a in t_file['image'].values()],0)
def find_zero(x):

    return np.argmin(np.abs(x))

def pick_rand(x):

    return np.random.choice(range(x.shape[0]))

func_list = [np.argmin, np.argmax, find_zero, pick_rand]

fig, m_axs = plt.subplots(2, len(func_list), figsize = (12, 6))

for c_func, n_axs in zip(func_list, m_axs.T):

    for col_idx, (c_ax, ax_name) in enumerate(zip(n_axs, 'XY')):

        show_idx = c_func(look_vec[:,col_idx])

        c_ax.imshow(syn_image_stack[show_idx], cmap = 'bone', interpolation='lanczos')

        c_ax.set_title('{} on {}\n {}'.format(c_func.__name__, 

                                              ax_name, 

                                              look_vec[show_idx, col_idx]))

        c_ax.axis('off')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(syn_image_stack/255.0, look_vec[:,:2], 

                                                    train_size = 0.5)

print('Train X size', X_train.shape, 'Test X size', X_test.shape, 'Mean', X_train.mean())

print('Train y size', y_train.shape, 'Test y size', y_test.shape, 'Mean', y_train.mean())
%%time

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer

full_pipe = Pipeline([('flatten', FunctionTransformer(lambda x: x.reshape(x.shape[0], -1), 

                                                      validate = False)), 

                      ('RF', RandomForestRegressor())])

full_pipe.fit(X_train[::5], y_train[::5])
y_pred = full_pipe.predict(X_test)

print('RF MSE values:', np.mean(np.square(y_pred-y_test),0))

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(y_pred[:,0], y_test[:, 0])

ax1.set_title('X Direction')

ax2.scatter(y_pred[:,1], y_test[:, 1])

ax2.set_title('Y Direction')

from keras.layers import Conv2D, Flatten, Dense, MaxPool2D

from keras.models import Sequential

simple_model = Sequential()

simple_model.add(Conv2D(input_shape = X_train.shape[1:]+(1,), filters = 8, kernel_size = (3,3)))

simple_model.add(MaxPool2D())

simple_model.add(Conv2D(filters = 16, kernel_size = (3,3)))

simple_model.add(MaxPool2D())

simple_model.add(Conv2D(filters = 32, kernel_size = (3,3)))

simple_model.add(MaxPool2D())

simple_model.add(Flatten())

simple_model.add(Dense(16, activation = 'relu'))

simple_model.add(Dense(2, activation = 'tanh'))

simple_model.compile('sgd', 'mse')

%simple_model.summary()
%%time

simple_model.fit(np.expand_dims(X_train, -1), y_train, epochs = 5)
y_pred = simple_model.predict(np.expand_dims(X_test,-1))

print('CNN MSE values:', np.mean(np.square(y_pred-y_test),0))

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(y_pred[:,0], y_test[:, 0])

ax1.set_title('X Direction')

ax2.scatter(y_pred[:,1], y_test[:, 1])

ax2.set_title('Y Direction')