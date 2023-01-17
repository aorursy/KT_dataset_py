import os
import h5py
import matplotlib.pyplot as plt
from skimage.util.montage import montage2d
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import numpy as np
import gc
gc.enable() # we come close to the memory limits and this seems to minimize kernel resets
montage3d = lambda x, **k: montage2d(np.stack([montage2d(y, **k) for y in x],0))
data_dir = '../input/mri-heart-processing/'
with h5py.File(os.path.join(data_dir, 'train_mri_128_128.h5'), 'r') as w:
    full_data = w['image'].value
    n_group = w['id'].value
    n_scalar = w['area_multiplier'].value
    y_target = w['systole'].value / n_scalar # remove the area scalar since we dont have this in the images
y_target.min(), y_target.max(), y_target.mean()
offset_value = 50
scale_factor = 100
y_target_class = ((y_target-offset_value)/scale_factor).clip(-1.5,1.5).reshape((-1,1))
_ = plt.hist(y_target_class)
y_target_class.shape
# instance normalization
safe_norm_func = lambda x: np.clip((x-x.mean())/(0.1+x.std()), -2, 2)
norm_ch_x_data = np.expand_dims(np.apply_along_axis(safe_norm_func, 0, full_data), -1) # add channels
del full_data
norm_ch_x_data.shape
fig, ax1 = plt.subplots(1,1, figsize = (8,8))
ax1.imshow(montage3d(norm_ch_x_data[np.random.choice(norm_ch_x_data.shape[0], size = 4), :, :, :, 0]))
%matplotlib inline
plt.hist(norm_ch_x_data[:5].ravel())
from keras.models import Sequential
from keras.layers import SpatialDropout3D, Dropout, Activation
from keras.layers import Conv3D, BatchNormalization, Dense, Flatten, Reshape, GlobalAveragePooling3D, MaxPooling3D
in_shape = norm_ch_x_data.shape[1:]
simple_model = Sequential()
simple_model.add(Conv3D(filters = 8, 
                        kernel_size = (3,1,1), 
                        input_shape = in_shape, 
                        activation = 'linear',
                        strides = (2, 1, 1),
                       use_bias = False))
simple_model.add(BatchNormalization())
simple_model.add(Activation('relu'))
simple_model.add(Conv3D(filters = 16, 
                        kernel_size = (3,1,1), 
                        input_shape = in_shape, 
                        activation = 'linear',
                        strides = (2, 1, 1),
                       use_bias = False))
simple_model.add(BatchNormalization())
simple_model.add(Activation('relu'))
simple_model.add(Conv3D(filters = 64, kernel_size = (3,3,3)))
simple_model.add(Conv3D(filters = 64, kernel_size = (1,3,3)))
simple_model.add(MaxPooling3D((1,2,2)))
simple_model.add(Conv3D(filters = 128, kernel_size = (1,3,3)))
simple_model.add(Conv3D(filters = 128, kernel_size = (1,3,3)))
simple_model.add(MaxPooling3D((1,2,2)))
simple_model.add(Conv3D(filters = 256, kernel_size = (1,3,3)))
simple_model.add(MaxPooling3D((1,2,2)))
simple_model.add(Conv3D(filters = 512, kernel_size = (1,3,3)))
simple_model.add(Conv3D(filters = 1024, kernel_size = (3,1,1)))
simple_model.add(SpatialDropout3D(0.5))
simple_model.add(GlobalAveragePooling3D())
simple_model.add(Dense(256))
simple_model.add(Dropout(0.5))
simple_model.add(Dense(y_target_class.shape[1], activation = 'tanh'))
simple_model.summary()
from keras.optimizers import Adam
simple_model.compile(loss = 'mse', 
                     optimizer = Adam(1e-4, decay = 1e-6), 
                     metrics = ['mae'])
loss_history = []
from sklearn.model_selection import train_test_split
# a simpler one is better here since the same patients are spread over multiple slices and we want to minimize leak without making too much hassle
def train_test_split(x, y, train_size, random_state):
    last_train_idx = int(train_size*x.shape[0])
    return x[:last_train_idx], x[last_train_idx+1:], y[:last_train_idx], y[last_train_idx+1:]
from keras.utils.np_utils import to_categorical
X_train, X_test, y_train, y_test = train_test_split(norm_ch_x_data, y_target_class, 
                                                   train_size = 0.7,
                                                   random_state = 2017)
del norm_ch_x_data
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('systole_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
loss_history += [simple_model.fit(X_train, y_train, 
          validation_data=(X_test, y_test),
                           shuffle = True,
                           batch_size = 32,
                           epochs = 30,
                                 callbacks = callbacks_list)]
simple_model.load_weights(weight_path)
simple_model.save('full_systolic_model.h5')
pred_test = simple_model.predict(X_test, verbose = 1)
fig, (ax1) = plt.subplots(1,1, figsize = (8, 8))
ax1.scatter(y_test, pred_test)
ax1.plot(y_test, y_test, 'r-')
for v, f in zip(simple_model.evaluate(X_test, y_test, verbose = 1), 
                simple_model.metrics_names):
    print('{}: difference - {:2.2f} ml'.format(f, scale_factor*v))
