from keras import layers, models

from keras.datasets import mnist

from keras.losses import mse, binary_crossentropy

from keras.utils import plot_model

from keras import backend as K

import numpy as np

import matplotlib.pyplot as plt

import argparse

from IPython.display import clear_output

from skimage.io import imsave

import os

import pandas as pd

import doctest

import copy

from tqdm import tqdm_notebook

# tests help notebooks stay managable



def autotest(func):

    globs = copy.copy(globals())

    globs.update({func.__name__: func})

    doctest.run_docstring_examples(

        func, globs, verbose=True, name=func.__name__)

    return func
%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

np.set_printoptions(precision=3)

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})
from sklearn.metrics import mutual_info_score

from scipy.ndimage import zoom

from scipy.stats import chi2_contingency





def calc_MI(x, y, bins=10):

    c_xy = np.histogram2d(x, y, bins)[0]

    try:

        g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")

        mi = 0.5 * g / c_xy.sum()

    except:

        mi = 0

    return mi





@autotest

def multiscale_image_mi(tensor_1,

                        tensor_2,

                        return_type='mean',

                        score_mode='calc'):

    """Calculate mutual information for images of different sizes by scaling them up

    >>> np.random.seed(0)

    >>> in_tensor = np.random.uniform(0, 1, size=(1, 64, 64, 1))

    >>> feat_tensor = np.concatenate([i*in_tensor[:, ::2, ::2] for i in range(4)], -1)

    >>> '%2.2f' % multiscale_image_mi(in_tensor, feat_tensor)

    '0.04'

    >>> '%2.2f' % multiscale_image_mi(in_tensor, feat_tensor, return_type='list')[0]

    '0.03'

    >>> '%2.2f' % multiscale_image_mi(in_tensor, np.random.uniform(-1, 0, size=(1, 64, 64, 3)))

    '0.01'

    """

    shape_1 = tensor_1.shape

    shape_2 = tensor_2.shape

    assert shape_1[0] == shape_2[0], "Batch size should be same"

    if score_mode == 'sklearn':

        mi_func = mutual_info_score

    elif score_mode == 'calc':

        def mi_func(x, y): return calc_MI(x, y)

    scaled_2 = zoom(

        tensor_2, [1, shape_1[1]/shape_2[1], shape_1[2]/shape_2[2], 1])

    out_mi = [(ch_1, ch_2, mi_func(

        scaled_2[:, :, :, ch_2].ravel(),

        tensor_1[:, :, :, ch_1].ravel()

    )

    )

        for ch_1 in range(shape_1[3])

        for ch_2 in range(shape_2[3])]

    if return_type == 'mean':

        return np.mean([mi_score for _, _, mi_score in out_mi])

    elif return_type == 'list_tuple':

        return out_mi

    elif return_type == 'list':

        return [mi_score for _, _, mi_score in out_mi]
from keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()



x_train = np.expand_dims(x_train.astype('float32') / 255, -1)

y_train = to_categorical(y_train)

x_test = np.expand_dims(x_test.astype('float32') / 255, -1)

y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)
from keras.optimizers import Adam

class_model = models.Sequential()

class_model.add(layers.Conv2D(4, kernel_size=(3,3), padding='same', activation='relu', input_shape=x_train.shape[1:]))

for i in range(3):

    class_model.add(layers.Conv2D(8*2**i, kernel_size=(3,3), padding='same', activation='relu'))

    class_model.add(layers.MaxPool2D((2, 2)))

class_model.add(layers.Flatten())

class_model.add(layers.Dense(128))

class_model.add(layers.Dense(y_train.shape[1], activation='softmax'))

class_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

class_model.summary()
def process_batch(in_model, in_batch, skip_last_layers=1):

    model_func = K.function(inputs=[in_model.inputs[0]],

              outputs=[x.get_output_at(0) for x in in_model.layers])

    layers_out = model_func([in_batch])

    model_out = layers_out[-1].reshape(in_batch.shape[0], 1, 1, -1)

    model_out = np.expand_dims(np.argmax(model_out, -1), -1)

    mi_scores = []

    for k, layer_val in enumerate(layers_out[:-skip_last_layers]):

        if layer_val.ndim==2:

            layer_val = np.expand_dims(layer_val, 1)

            layer_val = np.expand_dims(layer_val, 1)

        in_mi = multiscale_image_mi(in_batch, layer_val)

        out_mi = multiscale_image_mi(model_out, layer_val)

        mi_scores += [{'layer': k, 

                       'name': in_model.layers[k].name, 

                       'mi_to_input': in_mi, 

                       'mi_to_output': out_mi}]

    return mi_scores

pd.DataFrame(process_batch(class_model, x_train[0:32], skip_last_layers=3))
out_results = []

epoch_split = 4 # number of chunks to divide each epoch into

for i in tqdm_notebook(range(60)):

    out_results += [

        pd.DataFrame(process_batch(class_model, x_train[0:128], skip_last_layers=4)).\

            assign(epoch=i/epoch_split).\

            assign(test_accuracy=class_model.evaluate(x_test, y_test)[-1])

    ]

    c_idx = np.random.choice(range(x_train.shape[0]), size=x_train.shape[0]//epoch_split)

    class_model.fit(x_train[c_idx], y_train[c_idx], verbose=True, shuffle=True, epochs=1)
train_df = pd.concat(out_results).reset_index(drop=True)

train_df.to_csv('train_results.csv', index=False)

train_df.head(10)

fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

for c_name, c_rows in train_df.groupby(['layer', 'name']):

    d_rows = c_rows.sort_values('epoch')

    ax1.plot(d_rows['mi_to_input'].values, d_rows['mi_to_output'].values, '.-', label='{}: {}'.format(*c_name))

    ax1.plot(d_rows['mi_to_input'].iloc[0], d_rows['mi_to_output'].iloc[0], 'ks')

ax1.legend()

ax1.set_xlabel('$I(X; T)$')

ax1.set_ylabel('$I(Y; T)$');