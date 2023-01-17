%matplotlib inline
from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import zipfile as zf
import numpy as np
import h5py
from keras.utils.io_utils import HDF5Matrix
base_h5_dir = '../input/deeplesion-overview/'
base_img_dir = '../input/nih-deeplesion-subset/'
h5_path = os.path.join(base_h5_dir, 'deeplesion.h5')
with h5py.File(h5_path, 'r') as h:
    print(list(h.keys()))
    for k in ['image', 'mask']:
        print(k, h[k].shape, h[k].dtype)
    base_shape = h['image'].shape
ct_window_func = lambda x: np.clip((x+175.0)/275, -1, 1)
get_xyf = lambda s, e: (HDF5Matrix(h5_path, 'image', start=s, end=e, normalizer=ct_window_func), 
                       HDF5Matrix(h5_path, 'mask', start=s, end=e),
                       HDF5Matrix(h5_path, 'file_name', start=s, end=e)
                      )
train_split = 0.7
cut_val = int(base_shape[0]*train_split)
train_x, train_y, train_paths = get_xyf(0, cut_val)
test_x, test_y, test_paths = get_xyf(cut_val, None)
print(train_x.shape, test_x.shape)
from skimage.util.montage import montage2d as montage
t_x, t_y = train_x[:8], train_y[:8]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage(t_x[:, :, :, 0]), cmap = 'gray')
ax2.imshow(montage(t_y[:, :, :, 0]))
# we dont need full resolution images so we can just use a downsampled version
class DownsampledHDF5Matrix(HDF5Matrix):
    def __init__(self, datapath, dataset, downscale_factor, start=0, end=None, normalizer=None):
        ds_func = lambda x: x[:, ::downscale_factor, ::downscale_factor, :]
        ds_norm = ds_func if normalizer is None else lambda x: ds_func(normalizer(x))
        self.downscale_factor = downscale_factor
        super(DownsampledHDF5Matrix, self).__init__(datapath, dataset, start=start, end=end, normalizer=ds_norm)
        t_val = self[0:1]
        self._base_shape = t_val.shape[1:]
        self._base_dtype = t_val.dtype
    
    @property
    def shape(self):
        """Gets a numpy-style shape tuple giving the dataset dimensions.
        # Returns
            A numpy-style shape tuple. (self.data.shape[1]//self.downscale_factor, self.data.shape[2]//self.downscale_factor, self.data.shape[3])
        """
        return (self.end - self.start,) + self._base_shape
    
    @property
    def dtype(self):
        """Gets the datatype of the dataset.
        # Returns
            A numpy dtype string.
        """
        return self._base_dtype
ct_window_func = lambda x: np.clip((x+500.0)/600, -1, 1)
get_xyf = lambda s, e: (HDF5Matrix(h5_path, 'image', start=s, end=e, normalizer=ct_window_func), 
                       DownsampledHDF5Matrix(h5_path, 'mask', downscale_factor=4, start=s, end=e),
                       [x.decode() for x in HDF5Matrix(h5_path, 'file_name', start=s, end=e)[:]]
                      )
train_split = 0.7
cut_val = int(base_shape[0]*train_split)
train_x, train_y, train_paths = get_xyf(0, cut_val)
test_x, test_y, test_paths = get_xyf(cut_val, None)
print(test_x.shape, test_y.shape)
t_x, t_y = train_x[:8], train_y[:8]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage(t_x[:, :, :, 0]), cmap = 'gray')
ax2.imshow(montage(t_y[:, :, :, 0]))
patient_df = pd.read_csv(os.path.join(base_img_dir, 'DL_info.csv'))
patient_df = patient_df[patient_df['File_name'].isin(train_paths+test_paths)]
patient_df['training'] = patient_df['File_name'].isin(train_paths)
patient_df.sample(3)
from keras.layers import Input, Activation, Conv2D, MaxPool2D, UpSampling2D, Dropout, concatenate, BatchNormalization, Cropping2D, ZeroPadding2D, SpatialDropout2D
from keras.layers import Conv2DTranspose, Dropout, GaussianNoise
from keras.models import Model
from keras import backend as K

def up_scale(in_layer):
    filt_count = in_layer._keras_shape[-1]
    return Conv2DTranspose(filt_count//2+2, kernel_size = (2,2), strides = (2,2), padding = 'same')(in_layer)
def up_scale_fancy(in_layer):
    return UpSampling2D(size=(2,2))(in_layer)

input_layer = Input(shape=train_x.shape[1:])
sp_layer = GaussianNoise(0.1)(input_layer)
bn_layer = BatchNormalization()(sp_layer)
c1 = Conv2D(filters=8, kernel_size=(5,5), activation='relu', padding='same')(bn_layer)
l = MaxPool2D(strides=(2,2))(c1)
c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)

l = MaxPool2D(strides=(2,2))(c3)
c4 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)

l = SpatialDropout2D(0.25)(c4)
dil_layers = [l]
for i in [2, 4, 6, 8, 12, 18, 24]:
    dil_layers += [Conv2D(16,
                          kernel_size = (3, 3), 
                          dilation_rate = (i, i), 
                          padding = 'same',
                         activation = 'relu')(l)]
l = concatenate(dil_layers)

l = SpatialDropout2D(0.2)(concatenate([up_scale(l), c3], axis=-1))
l = Conv2D(filters=128, kernel_size=(2,2), activation='linear', padding='same')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = Conv2D(filters=96, kernel_size=(2,2), activation='relu', padding='same')(l)
l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
l = Conv2D(filters=16, kernel_size=(2,2), activation='linear', padding='same')(l)
l = Cropping2D((4,4))(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)

l = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
output_layer = ZeroPadding2D((4,4))(l)

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
seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
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
                  rotation_range = 7.5, 
                  width_shift_range = 0.02, 
                  height_shift_range = 0.02, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  brightness_range = [0.9, 1.1],
                  horizontal_flip = False, 
                  vertical_flip = False,
                  fill_mode = 'nearest',
                   data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)
dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)
def train_gen(batch_size = 16, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    while True:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        batch_count = train_x.shape[0]//batch_size
        batch_id = np.random.permutation(range(0, train_x.shape[0]-batch_size, batch_size))
        for c_idx in batch_id:
            g_x = image_gen.flow(train_x[c_idx:(c_idx+batch_size)], batch_size = batch_size, seed = seed, shuffle=True)
            g_y = label_gen.flow(train_y[c_idx:(c_idx+batch_size)], batch_size = batch_size, seed = seed, shuffle=True)
            yield next(g_x)/255.0, next(g_y)
cur_gen = train_gen(8)
t_x, t_y = next(cur_gen)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage(t_x[:, :, :, 0]), cmap='gray')
ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray_r')
batch_size = 16
loss_history = [seg_model.fit_generator(train_gen(batch_size), 
                             steps_per_epoch=train_x.shape[0]//batch_size, 
                             epochs=40, 
                             validation_data=(test_x, test_y),
                             callbacks=callbacks_list,
                            workers=2)]
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
for idx, (ax1, ax2, ax3) in enumerate(m_axs):
    ix = test_x[idx:idx+1]
    iy = test_y[idx:idx+idx+1]
    p_image = seg_model.predict(ix)
    ax1.imshow(ix[0,:,:,0], cmap = 'gray')
    ax1.set_title('Input Image')
    ax2.imshow(iy[0,:,:,0], vmin = 0, vmax = 1, cmap = 'bone_r' )
    ax2.set_title('Ground Truth')
    ax3.imshow(p_image[0,:,:,0], vmin = 0, vmax = 1, cmap = 'viridis' )
    ax3.set_title('Prediction')
