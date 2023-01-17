!ls -R ../input | head
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.util.montage import montage2d as montage
from skimage.io import imread
import os
base_dir = '../input'
from glob import glob
all_path_df = pd.DataFrame(dict(path = 
                                glob(os.path.join(base_dir,
                                                  'aorta-data', 'data',
                                                  '*', '*', '*', 'image.png'))))
all_path_df['patient_id'] = all_path_df['path'].map(lambda x: x.split('/')[-2])
all_path_df['train_group'] = all_path_df['path'].map(lambda x: x.split('/')[-4])
all_path_df['mask_path'] = all_path_df['path'].map(lambda x: x.replace('image.', 'mask.'))
all_path_df
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
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization
mod_in = Input((None, None, 1))
bn = BatchNormalization()(mod_in)
c1 = Conv2D(8, (3,3), padding = 'same')(bn)
c2 = Conv2D(8, (3,3), padding = 'same')(c1)
c_out = Conv2D(1, (1,1), padding = 'same', activation = 'sigmoid')(c2)
seg_model = Model(inputs = [mod_in], outputs = [c_out])
seg_model.summary()

import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 0.0*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
seg_model.compile(optimizer=Adam(1e-3, decay = 1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef", 
                      mode="max", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
loss_history = [seg_model.fit(train_X[0], train_X[1], 
                          batch_size = 8,
                          shuffle = True,
                        epochs = 50, 
                        validation_data = test_X,
                        callbacks = callbacks_list)]
def show_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich, np.concatenate(
            [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['true_positive_rate'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_true_positive_rate'] for mh in loss_history]),
                     'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('True Positive Rate\n(Positive Accuracy)')
    
    _ = ax3.plot(epich, np.concatenate(
        [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_binary_accuracy'] for mh in loss_history]),
                     'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('Binary Accuracy (%)')
    
    _ = ax4.plot(epich, np.concatenate(
        [mh.history['dice_coef'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_dice_coef'] for mh in loss_history]),
                     'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('DICE')
show_loss(loss_history)
seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')
fig, m_axs = plt.subplots(4,3, figsize = (20, 20))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for (ax1, ax2, ax3), ix, iy in zip(m_axs, test_X[0], test_X[1]):
    print(ix.shape, iy.shape)
    print(ix.min(), ix.max())
    print(iy.min(), iy.max())
    
    p_image = seg_model.predict(np.expand_dims(ix, 0))
    ax1.imshow(ix[:,:,0], cmap = 'bone')
    ax1.set_title('Input Image')
    ax2.imshow(iy[:,:,0], vmin = 0, vmax = 1, )
    ax2.set_title('Ground Truth')
    ax3.imshow(p_image[0,:,:,0], vmin = 0, vmax = 1)
    ax3.set_title('Prediction')
