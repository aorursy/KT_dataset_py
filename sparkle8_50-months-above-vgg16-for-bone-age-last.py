# copy the weights and configurations for the pre-trained models
!mkdir ~/.keras

!mkdir ~/.keras/models

!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/

!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/

!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # showing and rendering figures

# io related

from skimage.io import imread

import os

from glob import glob

# not needed in Kaggle, but required in Jupyter

%matplotlib inline 
base_bone_dir = os.path.join('..', 'input', 'rsna-bone-age')

age_df = pd.read_csv(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))

age_df['path'] = age_df['id'].map(lambda x: os.path.join(base_bone_dir,

                                                         'boneage-training-dataset', 

                                                         'boneage-training-dataset', 

                                                         '{}.png'.format(x)))

age_df['exists'] = age_df['path'].map(os.path.exists)

print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')

age_df['gender'] = age_df['male'].map(lambda x: 'male' if x else 'female')

boneage_mean = age_df['boneage'].mean()

boneage_div = 2*age_df['boneage'].std()

# we don't want normalization for now

boneage_mean = 0

boneage_div = 1.0

age_df['boneage_zscore'] = age_df['boneage'].map(lambda x: (x-boneage_mean)/boneage_div)

age_df.dropna(inplace = True)

age_df.sample(3)
age_df = age_df[age_df['boneage']>=60]

age_df = age_df[age_df['boneage']<=180]

age_df[['boneage', 'male', 'boneage_zscore']].hist(figsize = (10, 5))

age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)
from sklearn.model_selection import train_test_split

raw_train_df, valid_df = train_test_split(age_df, 

                                   test_size = 0.15, 

                                   random_state = 2018,

                                   stratify = age_df['boneage_category'])

print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
train_df = raw_train_df.groupby(['boneage_category', 'male']).apply(lambda x: x.sample(500, replace = True)

                                                      ).reset_index(drop = True)

print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])

train_df[['boneage', 'male']].hist(figsize = (10, 5))
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_resnet_v2 import preprocess_input

IMG_SIZE = (500, 500) # slightly smaller than vgg16 normally expects

core_idg = ImageDataGenerator(samplewise_center=False, 

                              samplewise_std_normalization=False, 

                              horizontal_flip = True, 

                              vertical_flip = False, 

                              height_shift_range = 0.2, 

                              width_shift_range = 0.2, 

                              rotation_range = 20, 

                              shear_range = 0.01,

                              fill_mode = 'nearest',

                              zoom_range=0.2,

                             preprocessing_function = preprocess_input)

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):

    base_dir = os.path.dirname(in_df[path_col].values[0])

    print('## Ignore next message from keras, values are replaced anyways')

    df_gen = img_data_gen.flow_from_directory(base_dir, 

                                     class_mode = 'sparse',

                                    **dflow_args)

    df_gen.filenames = in_df[path_col].values

    df_gen.classes = np.stack(in_df[y_col].values)

    df_gen.samples = in_df.shape[0]

    df_gen.n = in_df.shape[0]

    df_gen._set_index_array()

    df_gen.directory = '' # since we have the full path

    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))

    return df_gen
train_gen = flow_from_dataframe(core_idg, train_df, 

                             path_col = 'path',

                            y_col = 'boneage_zscore', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 32)



valid_gen = flow_from_dataframe(core_idg, valid_df, 

                             path_col = 'path',

                            y_col = 'boneage_zscore', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 128) # we can use much larger batches for evaluation

# used a fixed dataset for evaluating the algorithm

test_X, test_Y = next(flow_from_dataframe(core_idg, 

                               valid_df, 

                             path_col = 'path',

                            y_col = 'boneage_zscore', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 512)) # one big batch
t_x, t_y = next(train_gen)

fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))

for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):

    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)

    c_ax.set_title('%2.0f months' % (c_y*boneage_div+boneage_mean))

    c_ax.axis('off')
from keras.applications.inception_resnet_v2 import  InceptionResNetV2

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda

from keras.models import Model

in_lay = Input(t_x.shape[1:])

base_pretrained_model = InceptionResNetV2(input_shape =  t_x.shape[1:], include_top = False, weights = 'imagenet')

base_pretrained_model.trainable = False

pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]

pt_features = base_pretrained_model(in_lay)

from keras.layers import BatchNormalization

bn_features = BatchNormalization()(pt_features)



# here we do an attention mechanism to turn pixels in the GAP on an off



attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(bn_features)

attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)

attn_layer = LocallyConnected2D(1, 

                                kernel_size = (1,1), 

                                padding = 'valid', 

                                activation = 'sigmoid')(attn_layer)

# fan it out to all of the channels

up_c2_w = np.ones((1, 1, 1, pt_depth))

up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 

               activation = 'linear', use_bias = False, weights = [up_c2_w])

up_c2.trainable = False

attn_layer = up_c2(attn_layer)



mask_features = multiply([attn_layer, bn_features])

gap_features = GlobalAveragePooling2D()(mask_features)

gap_mask = GlobalAveragePooling2D()(attn_layer)

# to account for missing values from the attention model

gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])

gap_dr = Dropout(0.5)(gap)

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
bone_age_model.load_weights('../input/wights/bone_age_weights.best.hdf5')
bone_age_model.fit_generator(train_gen, 

                                  validation_data = (test_X, test_Y), 

                                  epochs = 15,

                                  callbacks = callbacks_list)
# load the best version of the model

bone_age_model.load_weights(weight_path)
from keras.models import load_model



bone_age_model.save('bone_age_model_new99.h5')  # creates a HDF5 file 'my_model.h5'

#del model  # deletes the existing model



# returns a compiled model

# identical to the previous one

#model1 = load_model('/kaggel/woking/my_model.h5')

#model1 = load_model('/kaggel/woking/my_model.h5')

#model = load_model('bone_age_model.h5')
# load the best version of the model

bone_age_model.load_weights(weight_path)
# get the attention layer since it is the only one with a single output dim

for attn_layer in bone_age_model.layers:

    c_shape = attn_layer.get_output_shape_at(0)

    if len(c_shape)==4:

        if c_shape[-1]==1:

            print(attn_layer)

            break
import keras.backend as K

rand_idx = np.random.choice(range(len(test_X)), size = 6)

attn_func = K.function(inputs = [bone_age_model.get_input_at(0), K.learning_phase()],

           outputs = [attn_layer.get_output_at(0)]

          )

fig, m_axs = plt.subplots(len(rand_idx), 2, figsize = (8, 4*len(rand_idx)))

[c_ax.axis('off') for c_ax in m_axs.flatten()]

for c_idx, (img_ax, attn_ax) in zip(rand_idx, m_axs):

    cur_img = test_X[c_idx:(c_idx+1)]

    attn_img = attn_func([cur_img, 0])[0]

    img_ax.imshow(cur_img[0,:,:,0], cmap = 'bone')

    attn_ax.imshow(attn_img[0, :, :, 0], cmap = 'viridis', 

                   vmin = 0, vmax = 1, 

                   interpolation = 'lanczos')

    real_age = boneage_div*test_Y[c_idx]+boneage_mean

    img_ax.set_title('Hand Image\nAge:%2.2fY' % (real_age/12))

    pred_age = boneage_div*bone_age_model.predict(cur_img)+boneage_mean

    attn_ax.set_title('Attention Map\nPred:%2.2fY' % (pred_age/12))

fig.savefig('attention_map.png', dpi = 300)
pred_Y = boneage_div*bone_age_model.predict(test_X, batch_size = 32, verbose = True)+boneage_mean

test_Y_months = boneage_div*test_Y+boneage_mean
fig, ax1 = plt.subplots(1,1, figsize = (6,6))

ax1.plot(test_Y_months, pred_Y, 'r.', label = 'predictions')

ax1.plot(test_Y_months, test_Y_months, 'b-', label = 'actual')

ax1.legend()

ax1.set_xlabel('Actual Age (Months)')

ax1.set_ylabel('Predicted Age (Months)')
ord_idx = np.argsort(test_Y)

ord_idx = ord_idx[np.linspace(0, len(ord_idx)-1, 8).astype(int)] # take 8 evenly spaced ones

fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))

for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):

    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')

    

    c_ax.set_title('Age: %2.1fY\nPredicted Age: %2.1fY' % (test_Y_months[idx]/12.0, 

                                                           pred_Y[idx]/12.0))

    c_ax.axis('off')

fig.savefig('trained_img_predictions.png', dpi = 300)
!rm -rf ~/.keras
#
