%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
from sklearn.model_selection import train_test_split
from skimage.util.montage import montage2d
from warnings import warn
def montage_nd(in_img):
    if len(in_img.shape)>3:
        return montage2d(np.stack([montage_nd(x_slice) for x_slice in in_img],0))
    elif len(in_img.shape)==3:
        return montage2d(in_img)
    else:
        warn('Input less than 3d image, returning original', RuntimeWarning)
        return in_img
BASE_IMG_PATH=os.path.join('..','input')
# show some of the files
all_images=glob(os.path.join(BASE_IMG_PATH,'3d_images','IMG_*'))
print(len(all_images),' matching files found:',all_images[0])
train_paths, test_paths = train_test_split(all_images, random_state = 2018, test_size = 0.5)
print(len(train_paths), 'training size')
print(len(test_paths), 'testing size')
DS_FACT = 8 # downscale
def read_all_slices(in_paths, rescale = True):
    cur_vol = np.expand_dims(np.concatenate([nib.load(c_path).get_data()[:, ::DS_FACT, ::DS_FACT] 
                                          for c_path in in_paths], 0), -1)
    if rescale:
        return (cur_vol.astype(np.float32) + 500)/2000.0
    else:
        return cur_vol/255.0
def read_both(in_paths):
    in_vol = read_all_slices(in_paths)
    in_mask = read_all_slices(map(lambda x: x.replace('IMG_', 'MASK_'), in_paths), rescale = False)
    return in_vol, in_mask
train_vol, train_mask = read_both(train_paths)
test_vol, test_mask = read_both(test_paths)
print('train', train_vol.shape, 'mask', train_mask.shape)
print('test', test_vol.shape, 'mask', test_mask.shape)
plt.hist(train_vol.ravel(), np.linspace(-1, 1, 50));
def gen_chunk(in_img, in_mask, slice_count = 10, batch_size = 16):
    while True:
        img_batch = []
        mask_batch = []
        for _ in range(batch_size):
            s_idx = np.random.choice(range(in_img.shape[0]-slice_count))
            img_batch += [in_img[s_idx:(s_idx+slice_count)]]
            mask_batch += [in_mask[s_idx:(s_idx+slice_count)]]
        yield np.stack(img_batch, 0), np.stack(mask_batch, 0)
# training we use larger batch sizes with fewer slices
train_gen = gen_chunk(train_vol, train_mask)
# for validation we use smaller batches with more slices
valid_gen = gen_chunk(test_vol, test_mask, slice_count = 100, batch_size = 1)
x_out, y_out = next(train_gen)
print(x_out.shape, y_out.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage_nd(x_out[...,0]), cmap = 'bone')
ax1.set_title('In Batch')
ax2.imshow(montage_nd(y_out[...,0]))
ax2.set_title('Out Batch')
from keras.preprocessing.image import ImageDataGenerator
d_gen = ImageDataGenerator(rotation_range=15, 
                           width_shift_range=0.15, 
                           height_shift_range=0.15, 
                           shear_range=0.1, 
                           zoom_range=0.25, 
                           fill_mode='nearest',
                           horizontal_flip=True, 
                           vertical_flip=False)

def gen_aug_chunk(in_gen):
    for i, (x_img, y_img) in enumerate(in_gen):
        xy_block = np.concatenate([x_img, y_img], 1).swapaxes(1, 4)[:, 0]
        img_gen = d_gen.flow(xy_block, shuffle=True, seed=i, batch_size = x_img.shape[0])
        xy_scat = next(img_gen)
        # unblock
        xy_scat = np.expand_dims(xy_scat,1).swapaxes(1, 4)
        yield xy_scat[:, :xy_scat.shape[1]//2], xy_scat[:, xy_scat.shape[1]//2:]

train_aug_gen = gen_aug_chunk(train_gen)
x_out, y_out = next(train_aug_gen)
print(x_out.shape, y_out.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage_nd(x_out[...,0]), cmap = 'bone')
ax1.set_title('In Batch')
ax2.imshow(montage_nd(y_out[...,0]))
ax2.set_title('Out Batch');
from keras.layers import ConvLSTM2D, Bidirectional, BatchNormalization, Conv3D, Cropping3D, ZeroPadding3D, Activation, Input
from keras.layers import MaxPooling3D, UpSampling3D, Deconvolution3D, concatenate
from keras.models import Model
in_layer = Input((None, None, None, 1))
bn = BatchNormalization()(in_layer)
cn1 = Conv3D(8, 
             kernel_size = (1, 5, 5), 
             padding = 'same',
             activation = 'relu')(bn)
cn2 = Conv3D(8, 
             kernel_size = (3, 3, 3),
             padding = 'same',
             activation = 'linear')(cn1)
bn2 = Activation('relu')(BatchNormalization()(cn2))

dn1 = MaxPooling3D((2, 2, 2))(bn2)
cn3 = Conv3D(16, 
             kernel_size = (3, 3, 3),
             padding = 'same',
             activation = 'linear')(dn1)
bn3 = Activation('relu')(BatchNormalization()(cn3))

dn2 = MaxPooling3D((1, 2, 2))(bn3)
cn4 = Conv3D(32, 
             kernel_size = (3, 3, 3),
             padding = 'same',
             activation = 'linear')(dn2)
bn4 = Activation('relu')(BatchNormalization()(cn4))

up1 = Deconvolution3D(16, 
                      kernel_size = (3, 3, 3),
                      strides = (1, 2, 2),
                     padding = 'same')(bn4)

cat1 = concatenate([up1, bn3])

up2 = Deconvolution3D(8, 
                      kernel_size = (3, 3, 3),
                      strides = (2, 2, 2),
                     padding = 'same')(cat1)

pre_out = concatenate([up2, bn2])

pre_out = Conv3D(1, 
             kernel_size = (1, 1, 1), 
             padding = 'same',
             activation = 'sigmoid')(pre_out)

pre_out = Cropping3D((1, 2, 2))(pre_out) # avoid skewing boundaries
out = ZeroPadding3D((1, 2, 2))(pre_out)
sim_model = Model(inputs = [in_layer], outputs = [out])
sim_model.summary()
sim_model.predict(x_out).shape # ensure the model works and the result has the right size
sim_model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['binary_accuracy', 'mse'])
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('convlstm_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
sim_model.fit_generator(train_aug_gen, 
                        epochs=20,
                        steps_per_epoch = 100, 
                        validation_data = valid_gen, 
                        validation_steps=10,
                       callbacks = callbacks_list)
sim_model.load_weights(weight_path)
test_single_vol, test_single_mask = read_both(test_paths[0:1])
test_single_vol = test_single_vol[1:]
test_single_mask = test_single_mask[1:]
pred_seg = sim_model.predict(np.expand_dims(test_single_vol,0))[0]
from skimage.util.montage import montage2d
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 7))
ax1.imshow(np.max(test_single_vol[::-1, :, :, 0], 1), cmap = 'bone')
ax1.set_aspect(0.3)
ax2.imshow(np.sum(pred_seg[::-1, :, :, 0], 1), cmap = 'bone_r')
ax2.set_title('Prediction')
ax2.set_aspect(0.3)
ax3.imshow(np.sum(test_single_mask[::-1, :, :, 0], 1), cmap = 'bone_r')
ax3.set_title('Actual Lung Volume')
ax3.set_aspect(0.3)
fig.savefig('full_scan_prediction.png', dpi = 300)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 7))
ax1.imshow(montage2d(test_single_vol[::6, :, :, 0]), cmap = 'bone')
ax2.imshow(montage2d(pred_seg[::6, :, :, 0]), cmap = 'viridis')
ax2.set_title('Prediction')
ax3.imshow(montage2d(test_single_mask[::6, :, :, 0]), cmap = 'viridis')
ax3.set_title('Actual Mask')
fig.savefig('subsample_pred.png', dpi = 300)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 7))
bow_lung_idx = np.array([30, 50]+[100, 150])
ax1.imshow(montage2d(test_single_vol[bow_lung_idx, :, :, 0]), cmap = 'bone')
ax2.imshow(montage2d(pred_seg[bow_lung_idx, :, :, 0]), cmap = 'bone_r')
ax2.set_title('Prediction')
ax3.imshow(montage2d(test_single_mask[bow_lung_idx, :, :, 0]), cmap = 'bone_r')
ax3.set_title('Actual Mask')
fig.savefig('bowel_vs_lung.png', dpi = 200)
