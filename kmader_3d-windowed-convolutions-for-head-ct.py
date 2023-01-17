%matplotlib inline

import os

import numpy as np

import pandas as pd

from glob import glob

import matplotlib.pyplot as plt

from pydicom import read_file as read_dicom

import SimpleITK as sitk

base_dir = os.path.join('..', 'input', 'qureai-headct')

reads_dir = os.path.join('..', 'input', 'headctreads')
from warnings import warn

from keras import backend as K

from keras.layers import Activation, Conv3D, Conv2D, multiply

from keras.layers import Convolution2D, Convolution3D

from keras.layers import Input

from keras.models import Model



def make_window_weights(window_list, is3d=False, dim_order=None, verbose=False):

    # type: (List[Tuple[str, Tuple[float, float]]], bool) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

    """

    A function to convert a list of windows to a set of tensorflow/keras weights

    :param window_list:

    :param is3d:

    :return:

    Examples

    --------

    >>> (w1, w2), (w3, w4) = make_window_weights(_ct_windows, dim_order = 'tf')

    >>> [w1.shape, w2.shape, w3.shape, w4.shape]

    [(1, 1, 1, 6), (6,), (1, 1, 6, 6), (6,)]

    >>> (w1, w2), (w3, w4) = make_window_weights(_ct_windows, is3d = True, dim_order = 'tf')

    >>> [w1.shape, w2.shape, w3.shape, w4.shape]

    [(1, 1, 1, 1, 6), (6,), (1, 1, 1, 6, 6), (6,)]

    >>> (w1, w2), (w3, w4) = make_window_weights(_ct_windows, dim_order = 'th')

    >>> [w1.shape, w2.shape, w3.shape, w4.shape]

    [(6, 1, 1, 1), (6,), (6, 6, 1, 1), (6,)]

    >>> (w1, w2), (w3, w4) = make_window_weights(_ct_windows, is3d = True, dim_order = 'th')

    >>> [w1.shape, w2.shape, w3.shape, w4.shape]

    [(6, 1, 1, 1, 1), (6,), (6, 6, 1, 1, 1), (6,)]

    """

    if is3d:

        base_size = (1, 1, 1)

    else:

        base_size = (1, 1)



    if dim_order is None:

        dim_order = K.image_dim_ordering()

    if dim_order == 'tf':

        pass

    elif dim_order == 'th':

        pass

    else:

        raise ValueError('Dim ordering must be either tf (tensorflow) or th (theano): {}'.format(dim_order))



    neg_wind_slopes = np.zeros(base_size + (1, len(window_list)), dtype=np.float32)

    pos_wind_slopes = np.zeros(base_size + (len(window_list), len(window_list)), dtype=np.float32)

    neg_wind_offsets = np.zeros((len(window_list),), dtype=np.float32)

    pos_wind_offsets = np.zeros((len(window_list),), dtype=np.float32)

    """

    x = [-1024, 1024]

    layer_output = W*x + b

    neg_layer: W = -1, b = -max_val

    neg_layer output = -1*x - max_val -> max_val - x

    pos_layer: W = -1/w, b =

    pos_layer output = -1/w * (max_val - x) - 1/w (max_val+min_val)

    x/w - (max_val + k)/w

    """

    for i, (name, (center, width)) in enumerate(window_list):

        n_width = width / 2.0

        min_val = center - n_width

        max_val = center + n_width



        if is3d:

            neg_wind_slopes[:, :, :, :, i] = -1 / width

        else:

            neg_wind_slopes[:, :, :, i] = -1 / width

        neg_wind_offsets[i] = max_val / width



        if is3d:

            pos_wind_slopes[:, :, :, i, i] = -1

        else:

            pos_wind_slopes[:, :, i, i] = -1

        pos_wind_offsets[i] = -(min_val - max_val) / width



        if verbose: print(name, (min_val, '-', max_val), width)



    if dim_order == 'tf':

        pass

    elif dim_order == 'th':

        roll_func = lambda ty: np.rollaxis(np.rollaxis(ty, -2, 0), -1, 0)

        neg_wind_slopes = roll_func(neg_wind_slopes)

        pos_wind_slopes = roll_func(pos_wind_slopes)

        pass



    return ((neg_wind_slopes, neg_wind_offsets), (pos_wind_slopes, pos_wind_offsets))





def wind_net_2d(in_img, w_list, dim_order=None, suffix=''):

    # type: (np.ndarray, List[Tuple[str, Tuple[float, float]]]) -> keras.models.Model

    """

    A function for creating a 2D network for applying windows to an image

    :param in_img:

    :param w_list:

    :return:

    Examples

    ------

    >>> t_model = wind_net_2d(np.zeros((2, 3)), _ct_windows, dim_order = 'tf')

    >>> [(ilay.name, ilay.output_shape) for ilay in t_model.layers]

    [('RawImageInput', (None, 2, 3, 1)), ('Negative-Windows', (None, 2, 3, 6)), ('Positive-Windows', (None, 2, 3, 6))]

    >>> fit_img = (t_model.predict(np.linspace(-1000, 1000, num = 6).reshape((1,2,3,1)))[0]*100).astype(int)

    >>> [(name, sorted(np.unique(fit_img[:,:,i]))) for i, (name, _) in enumerate(_ct_windows)]

    [('Soft Tissue', [0, 90, 100]), ('Lung', [16, 50, 83, 100]), ('Bone', [0, 6, 33, 60, 86]), ('Liver', [0, 100]), ('Brain', [0, 100]), ('LungNodes', [0, 100])]

    >>> h_model = wind_net_2d(np.zeros((2, 3)), _ct_windows, dim_order = 'th')

    >>> [(ilay.name, ilay.output_shape) for ilay in h_model.layers]

    [('RawImageInput', (None, 1, 2, 3)), ('Negative-Windows', (None, 6, 2, 3)), ('Positive-Windows', (None, 6, 2, 3))]

    >>> hfit_img = (h_model.predict(np.linspace(-1000, 1000, num = 6).reshape((1, 1, 2, 3)))[0]*100).astype(int)

    >>> [(name, sorted(np.unique(hfit_img[i, :,:]))) for i, (name, _) in enumerate(_ct_windows)]

    [('Soft Tissue', [0, 90, 100]), ('Lung', [16, 50, 83, 100]), ('Bone', [0, 6, 33, 60, 86]), ('Liver', [0, 100]), ('Brain', [0, 100]), ('LungNodes', [0, 100])]

    """

    if dim_order is None:

        dim_order = K.image_dim_ordering()

    if dim_order == 'th':

        in_shape = (1,) + in_img.shape[:2]

    elif dim_order == 'tf':

        in_shape = in_img.shape[:2] + (1,)

    else:

        raise ValueError('Dim ordering must be either tf (tensorflow) or th (theano): {}'.format(dim_order))

    K.set_image_dim_ordering(dim_order)

    bonus_args = {}

    if KERAS_2:

        bonus_args['data_format'] = K.image_data_format()

    return wind_net_2d_custom(in_shape=in_shape,

                              w_list=w_list,

                              suffix=suffix, **bonus_args)





def wind_net_2d_custom(in_shape, w_list, suffix, **bonus_args):

    in_node = Input(shape=in_shape, name='RawImageInput{}'.format(suffix))

    neg_weights, pos_weights = make_window_weights(w_list, is3d=False,

                                                   dim_order='tf'  # as of keras 2 this is always tf

                                                   )



    neg_node = Convolution2D(filters=len(w_list), kernel_size=(1, 1),

                             name='Negative-Windows{}'.format(suffix), activation='relu', use_bias=True,

                             weights=neg_weights, **bonus_args)(in_node)

    pos_node = Convolution2D(filters=len(w_list), kernel_size=(1, 1),

                             name='Positive-Windows{}'.format(suffix), activation='relu', use_bias=True,

                             weights=pos_weights, **bonus_args)(neg_node)

    return Model(inputs=[in_node], outputs=[pos_node])





def wind_net_3d(in_img, w_list, dim_order=None, suffix=''):

    # type: (np.ndarray, List[Tuple[str, Tuple[float, float]]]) -> keras.models.Model

    """

    Make a 3D windowing network using 2 1x1x1 convolutional layers with the appropriate weights

    :param in_img:

    :param w_list:

    :return: a simple two layer network

    Examples

    ------

    >>> t_model = wind_net_3d(np.zeros((2, 3, 4)), _ct_windows, dim_order = 'tf')

    >>> [(ilay.name, ilay.output_shape) for ilay in t_model.layers]

    [('RawImageInput', (None, 2, 3, 4, 1)), ('Negative-Windows', (None, 2, 3, 4, 6)), ('Positive-Windows', (None, 2, 3, 4, 6))]

    >>> fit_img = (t_model.predict(np.linspace(-1000, 1000, num = 24).reshape((1,2,3,4,1)))[0]*100).astype(int)

    >>> [(name, fit_img[:,:,:,i].min(),fit_img[:,:,:,i].max()) for i, (name, _) in enumerate(_ct_windows)]

    [('Soft Tissue', 0, 100), ('Lung', 16, 100), ('Bone', 0, 86), ('Liver', 0, 100), ('Brain', 0, 100), ('LungNodes', 0, 100)]

    >>> th_model = wind_net_3d(np.zeros((2, 3, 4)), _ct_windows, dim_order = 'th')

    >>> [(ilay.name, ilay.output_shape) for ilay in th_model.layers]

    [('RawImageInput', (None, 1, 2, 3, 4)), ('Negative-Windows', (None, 6, 2, 3, 4)), ('Positive-Windows', (None, 6, 2, 3, 4))]

    """

    if in_img is None:

        in_shape = (None, None, None)

    else:

        in_shape = in_img.shape[:3]

    if dim_order is None:

        dim_order = K.image_dim_ordering()

    if dim_order == 'th':

        in_shape = (1,) + in_shape

    elif dim_order == 'tf':

        in_shape = in_shape + (1,)

    else:

        raise ValueError('Dim ordering must be either tf (tensorflow) or th (theano): {}'.format(dim_order))

    return wind_net_3d_custom(in_shape, w_list=w_list, suffix=suffix)





def wind_net_3d_custom(in_shape, w_list, suffix):

    """

    For custom sized windows

    :param in_shape:

    :param w_list:

    :return:

    """

    in_node = Input(shape=in_shape, name='RawImageInput{}'.format(suffix))

    neg_weights, pos_weights = make_window_weights(w_list, is3d=True,

                                                   dim_order='tf'  # as of keras 2 this is always tf

                                                   )



    neg_node = Convolution3D(filters=len(w_list),

                             kernel_size=(1, 1, 1),

                             name='Negative-Windows{}'.format(suffix), activation='relu', use_bias=True,

                             weights=neg_weights)(in_node)

    pos_node = Convolution3D(filters=len(w_list),

                             kernel_size=(1, 1, 1),

                             name='Positive-Windows{}'.format(suffix), activation='relu', use_bias=True,

                             weights=pos_weights)(neg_node)

    return Model(inputs=[in_node], outputs=[pos_node])
all_dicom_paths = glob(os.path.join(base_dir, '*', '*', '*', '*', '*'))

print(len(all_dicom_paths), 'dicom files')

dicom_df = pd.DataFrame(dict(path = all_dicom_paths))

dicom_df['SliceNumber'] = dicom_df['path'].map(lambda x: int(os.path.splitext(x.split('/')[-1])[0][2:]))

dicom_df['SeriesName'] = dicom_df['path'].map(lambda x: x.split('/')[-2])

dicom_df['StudyID'] = dicom_df['path'].map(lambda x: x.split('/')[-3])

dicom_df['PatientID'] = dicom_df['path'].map(lambda x: x.split('/')[-4].split(' ')[0])

dicom_df['PatSeries'] = dicom_df.apply(lambda x: '{PatientID}-{SeriesName}'.format(**x), 1)

dicom_df.sample(3)
small_scans = dicom_df.groupby('PatSeries').count().reset_index().query('SliceNumber<240')

dicom_df = dicom_df[dicom_df['PatSeries'].isin(small_scans['PatSeries'])]

print('Removed big scans', dicom_df.shape[0], 'remaining images')
read_overview_df = pd.read_csv(os.path.join(reads_dir, 'reads.csv'))

read_overview_df['PatientID'] = read_overview_df['name'].map(lambda x: x.replace('-', '')) 

read_overview_df.sample(2).T
from collections import OrderedDict

new_reads = []

for _, c_row in read_overview_df.iterrows():

    base_dict = OrderedDict(PatientID = c_row['PatientID'], Category = c_row['Category'])

    for reader in ['R1', 'R2', 'R3']:

        c_dict = base_dict.copy()

        c_dict['Reader'] = reader

        for k,v in c_row.items():

            if (reader+':') in k:

                c_dict[k.split(':')[-1]] = v

        new_reads += [c_dict]

new_reads_df = pd.DataFrame(new_reads)

new_reads_df.to_csv('formatted_reads.csv')

new_reads_df.sample(5)
avg_reads_df = new_reads_df.groupby(['PatientID', 'Category']).agg('mean').reset_index()

read_dicom_df = pd.merge(avg_reads_df, dicom_df, on = 'PatientID')

read_dicom_df['Bleed'] = read_dicom_df.apply(lambda x: np.clip(x['BleedLocation-Left']+x['BleedLocation-Right']+x['ChronicBleed'], 0, 1), 1)

print(read_dicom_df.shape[0], 'total weakly-labeled slices')

read_dicom_df.sample(3)
read_dicom_df['directory'] = read_dicom_df['path'].map(lambda x: os.path.split(x)[0])

dicom_dir_df = read_dicom_df.groupby(['directory']).agg('first').reset_index().drop(['path'], 1)

print(dicom_dir_df.shape[0])

dicom_dir_df.sample(2)
def read_dicom_folder(in_dir):

    series_reader = sitk.ImageSeriesReader()

    # series_reader.LoadPrivateTagsOn()

    dicom_names = series_reader.GetGDCMSeriesFileNames(in_dir)

    series_reader.SetFileNames(dicom_names)

    out_img = series_reader.Execute()

    return sitk.GetArrayFromImage(out_img)
fig, m_axs = plt.subplots(3, 3, figsize = (20, 20))

for c_ax, (_, c_row) in zip(m_axs.flatten(), dicom_dir_df.groupby(['Bleed', 'Fracture']).apply(lambda x: x.sample(1)).reset_index(drop=True).iterrows()):

    try:

        c_img = read_dicom_folder(c_row['directory'])

        c_slice = np.mean(c_img, 0)

        c_ax.imshow(c_slice, cmap = 'bone')

        c_ax.set_title('Bleed: {Bleed:2.2f}, Fracture: {Fracture:2.2f}\n{SeriesName}'.format(**c_row))

    except Exception as e:

        c_ax.set_title('{}'.format(str(e)[:40]))

        print(e)

    c_ax.axis('off')
from sklearn.model_selection import train_test_split

valid_df = dicom_dir_df[['PatientID', 'Bleed']].drop_duplicates()

print('Patients', valid_df.shape[0])

train_ids, test_ids = train_test_split(valid_df[['PatientID']], 

                                       test_size = 0.25, 

                                       stratify = valid_df['Bleed'].map(lambda x: x>0))



train_unbalanced_df = dicom_dir_df[dicom_dir_df['PatientID'].isin(train_ids['PatientID'])]

test_df = dicom_dir_df[dicom_dir_df['PatientID'].isin(test_ids['PatientID'])]

print(train_unbalanced_df.shape[0], 'training images', test_df.shape[0], 'testing images')

train_unbalanced_df['Bleed'].hist(figsize = (10, 5))
train_df = train_unbalanced_df.groupby(train_unbalanced_df['Bleed'].map(lambda x: round(x*3)/3)).apply(lambda x: x.sample(200, replace = True)

                                                      ).reset_index(drop = True)

print('New Data Size:', train_df.shape[0], 'Old Size:', train_unbalanced_df.shape[0])

train_df['Bleed'].hist(figsize = (20, 5))
def ct_gen(in_df):

    while True:

        c_row = in_df.sample(1).iloc[0]

        c_image = read_dicom_folder(c_row['directory'])

        c_tensor = np.expand_dims(np.expand_dims(c_image, 0), -1)

        yield {'RawImageInput': c_tensor}, {'Bleed': np.reshape([c_row['Bleed']], (1, 1)), 'Fracture': np.reshape([c_row['Fracture']], (1, 1))}
train_gen = ct_gen(train_df)

x_vars, y_vars = next(train_gen)

for k,v in x_vars.items():

    print(k, v.shape)

for k,v in y_vars.items():

    print(k, v.shape)
ct_windows = [

    ('Soft Tissue', (40, 400)),

    ('Lung', (-600, 1200)),

    ('Bone', (450, 1500)),

    ('Liver', (90, 190)),

    ('Brain', (40, 80)),

    ('LungNodes', (-800, 400))

]

t_model = wind_net_3d(None, ct_windows, dim_order = 'tf')

t_model.trainable=False

t_model.summary()
wind_tensor = t_model.predict(x_vars)

test_tensor = x_vars['RawImageInput']

print(wind_tensor.shape)
fig, m_axs = plt.subplots(3, wind_tensor.shape[-1]+1, figsize = (25, 8))

for n_axs, c_slice in zip(m_axs, np.linspace(0, wind_tensor.shape[1], m_axs.shape[0]+2)[1:-1].astype(int)):

    n_axs[0].imshow(test_tensor[0, c_slice, :, :, 0], cmap='bone')

    n_axs[0].set_title('Raw Image')

    for i, c_ax in enumerate(n_axs[1:]):

        c_ax.imshow(wind_tensor[0, c_slice, :, :, i], cmap='bone')

        c_ax.axis('off')

        c_ax.set_title('Wind:{}'.format(i))
from keras import layers, models

in_ct_scan = layers.Input((None, 512, 512, 1), name='RawImageInput')

clean_scan = t_model(in_ct_scan)

x = clean_scan

for i in range(3):

    x = layers.Conv3D(8*2**i, (3, 3, 3), strides=(1, 2, 2), activation='relu', padding='same')(x)

image_features = layers.GlobalAveragePooling3D()(x)

image_features = layers.Dropout(0.5)(image_features)

dense_features = layers.Dense(64, activation='relu')(image_features)

bleed_out = layers.Dense(1, activation='sigmoid', name='Bleed')(dense_features)

fracture_out = layers.Dense(1, activation='sigmoid', name='Fracture')(dense_features)

bf_model = models.Model(inputs=[in_ct_scan], outputs=[bleed_out, fracture_out], name='BleedFractureModel')

bf_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',

                           metrics = ['binary_accuracy'])

bf_model.summary()
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))

for (c_x, c_y), c_ax in zip(train_gen, m_axs.flatten()):

    c_ax.imshow(c_x['RawImageInput'][0, 10, :, :, 0], cmap = 'bone')

    pred_y = bf_model.predict(c_x)

    c_ax.set_title(f"B:{c_y['Bleed'][0,0]:2.1%} F:{c_y['Fracture'][0,0]:2.1%}\nPred: B:{pred_y[0][0,0]:2.1%} F:{pred_y[1][0,0]:2.1%}")

    c_ax.axis('off')
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('cthead')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=6) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]
test_gen = ct_gen(test_df)
bf_model.fit_generator(ct_gen(train_df), 

                       steps_per_epoch = 50,

                        validation_data = test_gen, 

                       validation_steps = 50,

                              epochs = 20, 

                              callbacks = callbacks_list,

                             workers = 4,

                             use_multiprocessing=True, 

                             max_queue_size = 5

                            )
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))

for (c_x, c_y), c_ax in zip(test_gen, m_axs.flatten()):

    c_ax.imshow(c_x['RawImageInput'][0, 10, :, :, 0], cmap = 'bone')

    pred_y = bf_model.predict(c_x)

    c_ax.set_title(f"B:{c_y['Bleed'][0,0]:2.1%} F:{c_y['Fracture'][0,0]:2.1%}\nPred: B:{pred_y[0][0,0]:2.1%} F:{pred_y[1][0,0]:2.1%}")

    c_ax.axis('off')
out_vals = bf_model.evaluate_generator(test_gen, steps = 5, workers=1)

print(out_vals)
print('Accuracy Bleeds: %2.1f%%\nAccuracy Fractures: %2.1f%%' % (out_vals[-2]*100, out_vals[-1]*100))