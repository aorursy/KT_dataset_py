%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.util.montage import montage2d as montage
from skimage.io import imread
from glob import glob
import os
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
from sklearn.model_selection import train_test_split
train_valid_df = all_path_df.query('train_group=="train"')
train_df, valid_df = train_test_split(train_valid_df, test_size = 12, random_state = 2017)

test_df = all_path_df.query('train_group=="test"')
def df_to_block(in_df):
    return np.expand_dims(np.stack(in_df['image'], 0)/255.0, -1), np.expand_dims(np.stack(in_df['mask_image'], 0), -1)/255.0
train_X = df_to_block(train_df)
valid_X = df_to_block(valid_df)
test_X = df_to_block(test_df)
print(train_X[0].shape, valid_X[0].shape, test_X[0].shape)
from keras.layers import Input, Activation, Conv2D, MaxPool2D, UpSampling2D, Dropout, concatenate, BatchNormalization, Cropping2D, ZeroPadding2D, SpatialDropout2D
from keras.layers import Conv2DTranspose, Dropout, GaussianNoise
from keras.models import Model
from keras import backend as K
def up_scale(in_layer):
    filt_count = in_layer._keras_shape[-1]
    return Conv2DTranspose(filt_count//2+2, kernel_size = (3,3), strides = (2,2), padding = 'same')(in_layer)
def up_scale(in_layer):
    return UpSampling2D(size=(2,2))(in_layer)
input_layer = Input(shape=(None, None, 1))
sp_layer = GaussianNoise(0.05)(input_layer)
bn_layer = BatchNormalization()(sp_layer)
c1 = Conv2D(filters=8, kernel_size=(5,5), activation='relu', padding='same')(bn_layer)
l = MaxPool2D(strides=(2,2))(c1)
c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c3)
c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c4)
c5 = Conv2D(filters=64, kernel_size=(1,1), activation='linear', padding='same')(l)
c5 = BatchNormalization()(c5)
c5 = Activation('relu')(c5)
l = concatenate([up_scale(c5), c4], axis=-1)
l = Conv2D(filters=64, kernel_size=(2,2), activation='linear', padding='same')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = Dropout(0.2)(concatenate([up_scale(l), c3], axis=-1))
l = Conv2D(filters=32, kernel_size=(2,2), activation='linear', padding='same')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = Dropout(0.2)(concatenate([up_scale(l), c2], axis=-1))
l = Conv2D(filters=24, kernel_size=(2,2), activation='linear', padding='same')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = Dropout(0.2)(concatenate([up_scale(l), c1, bn_layer], axis=-1))
l = Conv2D(filters=32, kernel_size=(2,2), activation='linear', padding='same')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
dil_layers = [l]
for i in [2, 4, 6, 8, 12, 18, 24]:
    dil_layers += [Conv2D(16,
                          kernel_size = (3, 3), 
                          dilation_rate = (i, i), 
                          padding = 'same',
                         activation = 'relu')(l)]
l = concatenate(dil_layers)
l = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
l = Cropping2D((16,16))(l)
output_layer = ZeroPadding2D((16,16))(l)

seg_model = Model(input_layer, output_layer)
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
from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 5, 
                  width_shift_range = 0.05, 
                  height_shift_range = 0.05, 
                  shear_range = 0.025,
                  zoom_range = [0.9, 2.5],  
                  horizontal_flip = True, 
                  vertical_flip = False,
                  fill_mode = 'nearest',
                   data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)
def train_gen(batch_size = 16):
    seed = np.random.choice(range(9999))
    # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
    g_x = image_gen.flow(train_X[0], batch_size = batch_size, seed = seed)
    g_y = image_gen.flow(train_X[1], batch_size = batch_size, seed = seed)
    for i_x, i_y in zip(g_x, g_y):
        yield i_x, i_y
cur_gen = train_gen()
t_x, t_y = next(cur_gen)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage(t_x[:, :, :, 0]), cmap = 'bone')
ax2.imshow(montage(t_y[:, :, :, 0]))
loss_history = [seg_model.fit_generator(train_gen(32), 
                                         steps_per_epoch=50, 
                                         epochs = 30, 
                                         validation_data = valid_X,
                                         callbacks = callbacks_list,
                                        workers = 2
                                       )]
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
    p_image = seg_model.predict(np.expand_dims(ix, 0))
    ax1.imshow(ix[:,:,0], cmap = 'bone')
    ax1.set_title('Input Image')
    ax2.imshow(iy[:,:,0], vmin = 0, vmax = 1, cmap = 'bone_r' )
    ax2.set_title('Ground Truth')
    ax3.imshow(p_image[0,:,:,0], vmin = 0, vmax = 1, cmap = 'bone_r' )
    ax3.set_title('Prediction')
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
fig, m_axs = plt.subplots(2, 4, figsize = (30, 10))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for idx, (ax1, ix, iy) in enumerate(zip(m_axs.flatten(), valid_X[0], valid_X[1])):
    x_img = ix[:,:,0]
    p_image = seg_model.predict(np.expand_dims(ix, 0))[0, :, :, 0]
    gt_image = iy[:,:,0]
    rgb_img = label2rgb(image = np.clip(255*x_img, 0 , 255).astype(np.uint8), 
                        label = gt_image>0.5, 
                        bg_label = 0)
    rgb_img = mark_boundaries(rgb_img, p_image>0.5)
    ax1.imshow(rgb_img, cmap = 'bone')
    if idx==0:
        ax1.plot(0, 0, 'r-', label = 'Ground Truth')
        ax1.plot(0, 0, 'y-', label = 'Prediction')
        ax1.legend()
        ax1.set_title('Valid Set')
fig.savefig('model_valid_predictions.png', dpi = 300)
fig, m_axs = plt.subplots(2, 4, figsize = (30, 10))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for idx, (ax1, ix, iy) in enumerate(zip(m_axs.flatten(), test_X[0], test_X[1])):
    x_img = ix[:,:,0]
    p_image = seg_model.predict(np.expand_dims(ix, 0))[0, :, :, 0]
    gt_image = iy[:,:,0]
    rgb_img = label2rgb(image = np.clip(255*x_img, 0 , 255).astype(np.uint8), 
                        label = gt_image>0.5, 
                        bg_label = 0)
    rgb_img = mark_boundaries(rgb_img, p_image>0.5)
    ax1.imshow(rgb_img, cmap = 'bone')
    if idx==0:
        ax1.plot(0, 0, 'r-', label = 'Ground Truth')
        ax1.plot(0, 0, 'y-', label = 'Prediction')
        ax1.legend()
        ax1.set_title('Test Set')
fig.savefig('model_test_predictions.png', dpi = 300)
pd.DataFrame([{'metric': i, 
  'train': k_tr,
  'valid': k_v,
  'z_test': k_t} for i, k_tr, k_t, k_v in zip(seg_model.metrics_names, 
                       seg_model.evaluate(train_X[0], train_X[1], verbose = False),
                seg_model.evaluate(test_X[0], test_X[1], verbose = False),
                seg_model.evaluate(valid_X[0], valid_X[1], verbose = False))])

