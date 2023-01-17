%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.util.montage import montage2d as montage
from skimage.io import imread
import os
from glob import glob
base_dir = '../input'
all_path_df = pd.DataFrame(dict(path = 
                                glob(os.path.join(base_dir,
                                                  'aorta-data', 'data',
                                                  '*', '*', '*', 'image.png'))))
all_path_df['patient_id'] = all_path_df['path'].map(lambda x: x.split('/')[-2])
all_path_df['train_group'] = all_path_df['path'].map(lambda x: x.split('/')[-4])
all_path_df['mask_path'] = all_path_df['path'].map(lambda x: x.replace('image.', 'mask.'))
all_path_df.sample(5)
t_img = imread(all_path_df['path'].values[0])
t_mask = imread(all_path_df['mask_path'].values[0])
print(t_img.shape, t_img.min(), t_img.max(), t_img.mean())
print(t_mask.shape, t_mask.min(), t_mask.max())
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(t_img)
ax2.imshow(t_mask)
all_path_df['mask_image'] = all_path_df['mask_path'].map(lambda x: imread(x)[:, :, 0])
all_path_df['image'] = all_path_df['path'].map(lambda x: imread(x)[:, :, 0])
def pad_nd_image(in_img,  # type: np.ndarray
                 out_shape,  # type: List[Optional[int]]
                 mode='reflect',
                 **kwargs):
    # type: (...) -> np.ndarray
    """
    Pads an array to a specific size
    :param in_img:
    :param out_shape: the desired outputs shape
    :param mode: the mode to use in numpy.pad
    :param kwargs: arguments for numpy.pad
    :return:
    >>> pprint(pad_nd_image(np.eye(3), [7,7]))
    [[ 1.  0.  0.  0.  1.  0.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 0.  0.  1.  0.  0.  0.  1.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 1.  0.  0.  0.  1.  0.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 0.  0.  1.  0.  0.  0.  1.]]
    >>> pprint(pad_nd_image(np.eye(3), [2,2])) # should return the same
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    >>> t_mat = np.ones((2, 27, 29, 3))
    >>> o_img = pad_nd_image(t_mat, [None, 32, 32, None], mode = 'constant', constant_values=0)
    >>> o_img.shape
    (2, 32, 32, 3)
    >>> pprint(o_img.mean())
    0.7646484375
    >>> pprint(o_img[0,3,:,0])
    [ 0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
      1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.]
    """
    pad_dims = []
    for c_shape, d_shape in zip(in_img.shape, out_shape):
        pad_before, pad_after = 0, 0
        if d_shape is not None:
            if c_shape < d_shape:
                dim_diff = d_shape - c_shape
                pad_before = dim_diff // 2
                pad_after = dim_diff - pad_before
        pad_dims += [(pad_before, pad_after)]
    return np.pad(in_img, pad_dims, mode=mode, **kwargs)

def force_array_dim(in_img,  # type: np.ndarray
                    out_shape,  # type: List[Optional[int]]
                    pad_mode='reflect',
                    crop_mode='center',
                    **pad_args):
    # type: (...) -> np.ndarray
    """
    force the dimensions of an array by using cropping and padding
    :param in_img:
    :param out_shape:
    :param pad_mode:
    :param crop_mode: center or random (default center since it is safer)
    :param pad_args:
    :return:
    >>> np.random.seed(2018)
    >>> pprint(force_array_dim(np.eye(3), [7,7], crop_mode = 'random'))
    [[ 1.  0.  0.  0.  1.  0.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 0.  0.  1.  0.  0.  0.  1.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 1.  0.  0.  0.  1.  0.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 0.  0.  1.  0.  0.  0.  1.]]
    >>> pprint(force_array_dim(np.eye(3), [2,2], crop_mode = 'center'))
    [[ 1.  0.]
     [ 0.  1.]]
    >>> pprint(force_array_dim(np.eye(3), [2,2], crop_mode = 'random'))
    [[ 1.  0.]
     [ 0.  1.]]
    >>> pprint(force_array_dim(np.eye(3), [2,2], crop_mode = 'random'))
    [[ 0.  0.]
     [ 1.  0.]]
    >>> get_error(force_array_dim, in_img = np.eye(3), out_shape = [2,2], crop_mode = 'junk')
    'Crop mode must be random or center: junk'
    >>> t_mat = np.ones((1, 7, 9, 3))
    >>> o_img = force_array_dim(t_mat, [None, 12, 12, None], pad_mode = 'constant', constant_values=0)
    >>> o_img.shape
    (1, 12, 12, 3)
    >>> pprint(o_img.mean())
    0.4375
    >>> pprint(o_img[0,3,:,0])
    [ 0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.]
    """
    assert crop_mode in ['random', 'center'], "Crop mode must be random or " \
                                              "center: {}".format(crop_mode)

    pad_image = pad_nd_image(in_img, out_shape, mode=pad_mode, **pad_args)
    crop_dims = []
    for c_shape, d_shape in zip(pad_image.shape, out_shape):
        cur_slice = slice(0, c_shape)  # default
        if d_shape is not None:
            assert d_shape <= c_shape, \
                "Padding command failed: {}>={} - {},{}".format(d_shape,
                                                                c_shape,
                                                                pad_image.shape,
                                                                out_shape
                                                                )
            if d_shape < c_shape:
                if crop_mode == 'random':
                    start_idx = np.random.choice(
                        range(0, c_shape - d_shape + 1))
                    cur_slice = slice(start_idx, start_idx + d_shape)
                else:
                    start_idx = (c_shape - d_shape) // 2
                    cur_slice = slice(start_idx, start_idx + d_shape)
        crop_dims += [cur_slice]
    return pad_image.__getitem__(crop_dims)
all_path_df['mask_image'] = all_path_df['mask_image'].map(lambda x: force_array_dim(x, (224, 224)))
all_path_df['image'] = all_path_df['image'].map(lambda x: force_array_dim(x, (224, 224)))
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage(np.stack(all_path_df['image'], 0)))
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage(np.stack(all_path_df['mask_image'], 0)))
train_df = all_path_df.query('train_group=="train"')
test_df = all_path_df.query('train_group=="test"')
def df_to_block(in_df):
    return np.expand_dims(np.stack(in_df['image'], 0)/255.0, -1), np.expand_dims(np.stack(in_df['mask_image'], 0), -1)/255.0
train_X = df_to_block(train_df)
test_X = df_to_block(test_df)
print(train_X[0].shape,test_X[0].shape)
D_FACTOR = 15
def img_to_xyr(in_img, norm = True):
    y_coord, x_coord = np.where(in_img[:, :, 0]>0)
    if len(y_coord)==0:
        return 0, 0, 0.5
    x_mean = x_coord.mean()
    y_mean = y_coord.mean()
    r = np.sqrt(np.square(x_coord-x_mean)+np.square(y_coord-y_mean))
    img_shape = in_img
    scale_fact = (np.mean(in_img.shape[0:2])/D_FACTOR)
    r_90 = np.percentile(r, 90)/scale_fact
    if norm:
        x_mean = (1.0*x_mean-in_img.shape[1]//2)/(in_img.shape[1]//2)
        y_mean = (1.0*y_mean-in_img.shape[0]//2)/(in_img.shape[0]//2)
    return x_mean, y_mean, r_90

def stack_to_xyr(in_stack):
    return np.stack([img_to_xyr(c_slice) for c_slice in in_stack], 0)

def xyr_to_img(in_xyr, img_shape):
    x_norm, y_norm, r_norm = in_xyr
    
    yy, xx = np.meshgrid(range(img_shape[0]), 
                         range(img_shape[1]), 
                         indexing = 'ij')
    out_img = np.zeros(xx.shape+(1,), dtype = np.float32)
    x_mean = x_norm*img_shape[1]//2+(img_shape[1]//2)
    y_mean = y_norm*img_shape[0]//2+(img_shape[0]//2)
    scale_fact = np.mean(img_shape[0:2])/D_FACTOR
    r_90 = r_norm*scale_fact
    
    out_img[np.sqrt(np.square(xx-x_mean)+np.square(yy-y_mean))<=r_90, :] = 1.0
    return out_img
# check to see if it works
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
t_img = train_X[1][0].copy()
ax1.imshow(t_img[:, :, 0], cmap = 'bone')
ax1.set_title('Original')

x_mean, y_mean, r_90 = img_to_xyr(t_img, norm = False)
circle = Circle((x_mean, y_mean), radius = r_90*224/D_FACTOR)
ax1.add_collection(PatchCollection([circle], alpha = 0.25, facecolor = 'red'))
x_norm, y_norm, r_90 = img_to_xyr(t_img, norm = True)
gen_img = xyr_to_img((x_norm, y_norm, r_90), t_img.shape)
ax2.imshow(gen_img[:, :, 0])
ax2.set_title('Reconstruction')
import seaborn as sns
train_xyr = stack_to_xyr(train_X[1])
test_xyr = stack_to_xyr(test_X[1])
sns.pairplot(pd.DataFrame(train_xyr))
from keras.layers import Input, Activation, Conv2D, MaxPool2D, Dropout, concatenate, BatchNormalization, Cropping2D, ZeroPadding2D, SpatialDropout2D
from keras.layers import Dropout, GaussianNoise, Flatten, Dense, add, LeakyReLU, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

input_layer = Input(shape=train_X[0].shape[1:])
sp_layer = GaussianNoise(0.05)(input_layer)
bn_layer = BatchNormalization()(sp_layer)
out_layers = []
def conv_block(in_layer, base_depth, downsample = False):
    x = Conv2D(filters=base_depth, kernel_size=(1,1), activation='relu', padding='same')(in_layer)
    x = Conv2D(filters=base_depth*2, kernel_size=(3,3), activation='relu', padding='same')(x)
    y = Conv2D(filters=base_depth*2, kernel_size=(1,1), activation='linear', padding='same')(in_layer)
    x = add([x, y])
    if downsample:
        x = Conv2D(filters=base_depth*2, kernel_size=(3,3), activation='linear', padding='same', strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    return x

c1 = conv_block(bn_layer, 8)
c2 = conv_block(c1, 16, downsample = True)

c3 = conv_block(c2, 32)
c4 = conv_block(c3, 32)
c5 = conv_block(c4, 32, downsample = True)

c6 = conv_block(c5, 64)
c7 = conv_block(c6, 64)
c8 = conv_block(c7, 64, downsample = True)

cb = conv_block(c8, 128)
cb = conv_block(cb, 128)
c11 = conv_block(cb, 128, downsample = True)

cb = conv_block(c11, 256)
cb = conv_block(cb, 256)
c12 = conv_block(cb, 256, downsample = True)


c_out = GlobalAveragePooling2D()(c12)
c_out = Dense(512)(c_out)
c_out = Dropout(0.2)(c_out)

xy_coord = Dense(32)(c_out)
xy_coord = Dense(2, activation = 'tanh', name = 'ToXYCoordinates')(xy_coord)
r_coord = Dense(32)(c_out)
r_coord = Dense(1, activation = 'sigmoid', name = 'ToRadius')(r_coord)
output_layer = concatenate([xy_coord, r_coord])
circ_model = Model(input_layer, output_layer)
circ_model.summary()
def show_circle_results(frames = 4):
    fig, m_axs = plt.subplots(frames, 3, figsize = (20, 20))
    [c_ax.axis('off') for c_ax in m_axs.flatten()]
    for (ax1, ax2, ax3), ix, iy in zip(m_axs, test_X[0], test_X[1]):
        print(ix.shape, iy.shape)
        print(ix.min(), ix.max())
        print(iy.min(), iy.max())

        p_xyr = circ_model.predict(np.expand_dims(ix, 0))[0]
        p_image = xyr_to_img(p_xyr, ix.shape)
        ax1.imshow(ix[:,:,0], cmap = 'bone')
        ax1.set_title('Input Image')
        ax2.imshow(iy[:,:,0], vmin = 0, vmax = 1, )
        ax2.set_title('Ground Truth')
        ax3.imshow(p_image[:,:,0], vmin = 0, vmax = 1)
        ax3.set_title('Prediction {}'.format(p_xyr))
show_circle_results(2)
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, mean_squared_error
def center_dist(y_true, y_pred):
    return K.sqrt(K.square(y_true[:, 0]-y_pred[:, 0])+K.square(y_true[:, 1]-y_pred[:, 1]))

def area_mismatch(y_true, y_pred):
    return mean_squared_error(y_true[:, 2:3], y_pred[:, 2:3])
# TODO: derive real IoU formula
def cent_and_area(y_true, y_pred):
    return center_dist(y_true, y_pred)+0.5*area_mismatch(y_true, y_pred)

circ_model.compile(optimizer=Adam(1e-4, decay = 1e-6), loss=cent_and_area, metrics=[center_dist, area_mismatch])
loss_history = [circ_model.fit(train_X[0], train_xyr, 
                          batch_size = 16,
                          shuffle = True,
                        epochs = 10, 
                        validation_data = (test_X[0], test_xyr))]
circ_model.save('seg_model.h5')
show_circle_results(2)
from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 30, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.1,
                  zoom_range = [0.8, 1.2],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')
image_gen = ImageDataGenerator(**dg_args)
def train_gen():
    np.random.seed(2017)
    
    while True:
        c_seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(train_X[0], batch_size = 32, seed = c_seed, shuffle = True)
        g_y = image_gen.flow(train_X[1], batch_size = 32, seed = c_seed, shuffle = True)
        for i_x, i_y in zip(g_x, g_y):
            yield i_x, stack_to_xyr(i_y)
cur_gen = train_gen()
t_x, t_y = next(cur_gen)
print(t_y.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage(t_x[:, :, :, 0]), cmap = 'bone')
ax2.plot(t_y)
fig, m_axs = plt.subplots(1, t_y.shape[1], figsize = (30, 10))
for _, (t_x, t_y) in zip(range(10), cur_gen):
    pred_y = circ_model.predict(t_x)
    for i, c_ax in enumerate(m_axs):
        c_ax.scatter(pred_y[:, i], t_y[:, i])
        c_ax.plot(t_y[:, i], t_y[:, i], '-')
for i, c_ax in enumerate(m_axs):
    c_ax.axis('equal')
    c_ax.set_xlim(-1, 1)
    c_ax.set_ylim(-1, 1)
    c_ax.set_xlabel('Predicted')
    c_ax.set_ylabel('Actual Value')
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('circle_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
loss_history += [circ_model.fit_generator(cur_gen, 
                                         steps_per_epoch=100, 
                                         epochs = 30, 
                                         validation_data = [test_X[0], test_xyr],
                                         callbacks = callbacks_list)]
circ_model.load_weights(weight_path)
circ_model.save('seg_model.h5')
show_circle_results(4)
fig, m_axs = plt.subplots(2, t_y.shape[1], figsize = (30, 20))
for _, (t_x, t_y) in zip(range(10), cur_gen):
    pred_y = circ_model.predict(t_x)
    for i, (c_ax, d_ax) in enumerate(m_axs.T):
        c_ax.scatter(pred_y[:, i], t_y[:, i])
        c_ax.plot(t_y[:, i], t_y[:, i], '-')
        d_ax.hist(t_y[:, i], color = 'red', alpha = 0.5)
        d_ax.hist(pred_y[:, i], color = 'blue', alpha = 0.5)
for i, (c_ax, d_ax) in enumerate(m_axs.T):
    c_ax.axis('equal')
    c_ax.set_xlim(-1, 1)
    c_ax.set_ylim(-1, 1)
    c_ax.set_xlabel('Predicted')
    c_ax.set_ylabel('Actual Value')
