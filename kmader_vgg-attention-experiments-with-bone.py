import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # showing and rendering figures

# io related

from skimage.io import imread

import os

from glob import glob

from keras.utils.io_utils import HDF5Matrix

# not needed in Kaggle, but required in Jupyter

%matplotlib inline 
X_train = HDF5Matrix(os.path.join('..', 'input', 'train.h5'), 'feature_vec')

X_test = HDF5Matrix(os.path.join('..', 'input', 'test.h5'), 'feature_vec')

print(X_train.shape, X_test.shape)
Y_train_raw = HDF5Matrix(os.path.join('..', 'input', 'train.h5'), 'boneage')[:]

Y_test_raw = HDF5Matrix(os.path.join('..', 'input', 'test.h5'), 'boneage')[:]

boneage_mean = Y_train_raw.mean()

boneage_div = 2*Y_train_raw.std()

boneage_mean = 0

boneage_div = 1.0

Y_train = (Y_train_raw-boneage_mean)/boneage_div

Y_test = (Y_test_raw-boneage_mean)/boneage_div

print(Y_train.shape, Y_test.shape)

print('train', Y_train.mean(), Y_train.max())
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda, BatchNormalization

from keras.models import Model

in_lay = Input(X_train.shape[1:])

pt_depth = X_train.shape[-1]



bn_lay = BatchNormalization()(in_lay)

attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(bn_lay)

attn_layer = Conv2D(32, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)

attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)

attn_layer = LocallyConnected2D(1, kernel_size = (1,1), padding = 'valid', activation = 'sigmoid')(attn_layer)

# fan it out to all of the channels

up_c2_w = np.ones((1, 1, 1, pt_depth))

up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 

               activation = 'linear', use_bias = False, weights = [up_c2_w])

up_c2.trainable = False

attn_layer = up_c2(attn_layer)



mask_features = multiply([attn_layer, in_lay])

gap_features = GlobalAveragePooling2D()(mask_features)

gap_mask = GlobalAveragePooling2D()(attn_layer)

# to account for missing values from the attention model

gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])

gap_dr = BatchNormalization()(Dropout(0.5)(gap))

dr_steps = Dropout(0.25)(Dense(1024, activation = 'elu')(gap_dr))

out_layer = Dense(1, activation = 'linear')(dr_steps) # linear is what 16bit did

bone_age_model = Model(inputs = [in_lay], outputs = [out_layer])



from keras.metrics import mean_absolute_error

def mae_months(in_gt, in_pred):

    return mean_absolute_error(boneage_div*in_gt, boneage_div*in_pred)



bone_age_model.compile(optimizer = 'adam', loss = 'mse',

                           metrics = [mae_months])



bone_age_model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('bone_age')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=5) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]
loss_history = bone_age_model.fit(X_train, Y_train, 

                    batch_size = 32,

                    validation_data = (X_test, Y_test),

                    epochs = 20,

                    shuffle = 'batch',

                    callbacks = callbacks_list)
fig, ax1 = plt.subplots(1,1, figsize = (6,6))

ax1.plot(loss_history.epoch,

         loss_history.history['mae_months'], 'r-', label = 'Training MAE')

ax1.plot(loss_history.epoch,

         loss_history.history['val_mae_months'], 'b-', label = 'Validation MAE')
# load the best version of the model

bone_age_model.load_weights(weight_path)
pred_Y = boneage_div*bone_age_model.predict(X_test, batch_size = 64, verbose = True)+boneage_mean

test_Y_months = boneage_div*Y_test+boneage_mean
fig, ax1 = plt.subplots(1,1, figsize = (6,6))

ax1.plot(test_Y_months, pred_Y, 'r.', label = 'predictions')

ax1.plot(test_Y_months, test_Y_months, 'b-', label = 'actual')

ax1.legend()

ax1.set_xlabel('Actual Age (Months)')

ax1.set_ylabel('Predicted Age (Months)')
from keras.applications.vgg16 import VGG16

from keras.models import Sequential

if os.path.exists('~/.keras'):

    base_pretrained_model = VGG16(input_shape =  (256, 256, 3), 

                                  include_top = False, weights = 'imagenet' 

                                 )

    base_pretrained_model.trainable = False

    out_model = Sequential()

    out_model.add(base_pretrained_model)

    out_model.add(bone_age_model)

    out_model.save('full_model.h5')

    out_model.summary()

else:

    print('Cannot Create Full Model without pretrained weights!')