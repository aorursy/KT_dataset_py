from keras.models import Model

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, BatchNormalization

from keras.layers import Input, Conv2D, multiply, LocallyConnected2D, Lambda, Flatten, concatenate

from keras.layers import GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras import optimizers

from keras.metrics import mean_absolute_error

from keras.applications import Xception

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import os

import matplotlib.pyplot as plt

import seaborn as sns
# hyperparameters

EPOCHS = 15

LEARNING_RATE = 0.0006

BATCH_SIZE_TRAIN = 32

BATCH_SIZE_VAL = 256



#image parameters 

PIXELS = 299 #default for Xception

CHANNELS = 3

IMG_SIZE = (PIXELS, PIXELS)

IMG_DIMS = (PIXELS, PIXELS, CHANNELS)

VALIDATION_FRACTION = 0.25

SEED = 7834
path = '../input/boneage-training-dataset/boneage-training-dataset/'

path = '../input/'

train_path = path + 'boneage-training-dataset/boneage-training-dataset/'

test_path = path + 'boneage-test-dataset/boneage-test-dataset/'



df = pd.read_csv(path + 'boneage-training-dataset.csv')

files = [train_path + str(i) + '.png' for i in df['id']]

df['file'] = files

df['exists'] = df['file'].map(os.path.exists)
fig, ax = plt.subplots()

ax = sns.distplot(df['boneage'], bins=10)

ax.set(xlabel='Boneage (months)', ylabel='Density',

    title='Boneage distribution');
boneage_mean = df['boneage'].mean()

boneage_div = 2 * df['boneage'].std()

df['boneage_zscore'] = df['boneage'].map(lambda x:

    (x - boneage_mean) / boneage_div)

df.dropna(inplace=True)



df['gender'] = df['male'].map(lambda x: 1 if x else 0)



df['boneage_category'] = pd.cut(df['boneage'], 10)

raw_train_df, raw_valid_df = train_test_split(df, test_size=VALIDATION_FRACTION,

  random_state=2018, stratify=df['boneage_category'])

train_df = raw_train_df.groupby(['boneage_category', 'male']).apply(

  lambda x: x.sample(500, replace=True)).reset_index(drop=True)

valid_df, test_df = train_test_split(raw_valid_df,

  test_size=VALIDATION_FRACTION, random_state=2019)
fig, ax = plt.subplots()

ax = sns.distplot(train_df['boneage'], bins=10)

ax.set(xlabel='Boneage (months)', ylabel='Density',

    title='Boneage training distribution');
optim = optimizers.Nadam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.0003)
weight_path = "{}_weights.best.hdf5".format('bone_age3')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,

    save_best_only=True, mode='min', save_weights_only=True)



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8,

    patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=5,

    min_lr=0.00006)

early = EarlyStopping(monitor="val_loss", mode="min", patience=8)

callbacks_list = [checkpoint, early, reduceLROnPlat]
BATCH_SIZE_TEST = len(test_df) // 3

STEP_SIZE_TEST = 3

STEP_SIZE_TRAIN = len(train_df) // BATCH_SIZE_TRAIN

STEP_SIZE_VALID = len(valid_df) // BATCH_SIZE_VAL
def gen_2inputs(imgDatGen, df, batch_size, seed, img_size):

    gen_img = imgDatGen.flow_from_dataframe(dataframe=df,

        x_col='file', y_col='boneage_zscore',

        batch_size=batch_size, seed=seed, shuffle=True, class_mode='other',

        target_size=img_size, color_mode='rgb',

        drop_duplicates=False)

    

    gen_gender = imgDatGen.flow_from_dataframe(dataframe=df,

        x_col='file', y_col='gender',

        batch_size=batch_size, seed=seed, shuffle=True, class_mode='other',

        target_size=img_size, color_mode='rgb',

        drop_duplicates=False)

    

    while True:

        X1i = gen_img.next()

        X2i = gen_gender.next()

        yield [X1i[0], X2i[1]], X1i[1]
def test_gen_2inputs(imgDatGen, df, batch_size, img_size):

    gen_img = imgDatGen.flow_from_dataframe(dataframe=df,

        x_col='file', y_col='boneage_zscore',

        batch_size=batch_size, shuffle=False, class_mode='other',

        target_size=img_size, color_mode='rgb',

        drop_duplicates=False)

    

    gen_gender = imgDatGen.flow_from_dataframe(dataframe=df,

        x_col='file', y_col='gender',

        batch_size=batch_size, shuffle=False, class_mode='other',

        target_size=img_size, color_mode='rgb',

        drop_duplicates=False)

    

    while True:

        X1i = gen_img.next()

        X2i = gen_gender.next()

        yield [X1i[0], X2i[1]], X1i[1]
train_idg = ImageDataGenerator(zoom_range=0.2,

                               fill_mode='nearest',

                               rotation_range=25,  

                               width_shift_range=0.25,  

                               height_shift_range=0.25,  

                               vertical_flip=False, 

                               horizontal_flip=True,

                               shear_range = 0.2,

                               samplewise_center=False, 

                               samplewise_std_normalization=False)



val_idg = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)



train_flow = gen_2inputs(train_idg, train_df, BATCH_SIZE_TRAIN, SEED, IMG_SIZE)



valid_flow = gen_2inputs(val_idg, valid_df, BATCH_SIZE_VAL, SEED, IMG_SIZE)



test_idg = ImageDataGenerator()



test_flow = test_gen_2inputs(test_idg, test_df, 789, IMG_SIZE)
def mae_months(in_gt, in_pred):

    return mean_absolute_error(boneage_div * in_gt, boneage_div * in_pred)
# Two inputs. One for gender and one for images

in_layer_img = Input(shape=IMG_DIMS, name='input_img')

in_layer_gender = Input(shape=(1,), name='input_gender')



# Pretrained neural network

base = Xception(input_shape=IMG_DIMS, weights='imagenet', include_top=False)



pt_depth = base.get_output_shape_at(0)[-1]

pt_features = base(in_layer_img)

bn_features = BatchNormalization()(pt_features)



# Attention layer

attn_layer = Conv2D(64, kernel_size=(1,1), padding='same', activation='relu')(bn_features)

attn_layer = Conv2D(16, kernel_size=(1,1), padding='same', activation='relu')(attn_layer)

attn_layer = LocallyConnected2D(1, kernel_size=(1,1), padding='valid',

    activation = 'sigmoid')(attn_layer)



# Applying attention to all features coming out of bn_features

up_c2_w = np.ones((1, 1, 1, pt_depth))

up_c2 = Conv2D(pt_depth, kernel_size=(1,1), padding='same',

    activation='linear', use_bias=False, weights=[up_c2_w])

up_c2.trainable = False

attn_layer = up_c2(attn_layer)



mask_features = multiply([attn_layer, bn_features])



# Global Average Pooling 2D

gap_features = GlobalAveragePooling2D()(mask_features)

gap_mask = GlobalAveragePooling2D()(attn_layer)

gap = Lambda(lambda x: x[0]/x[1], name='RescaleGAP')([gap_features, gap_mask])

gap_dr = Dropout(0.5)(gap)

dr_steps = Dropout(0.25)(Dense(1024, activation = 'elu')(gap_dr))



# This is where gender enters in the model

feature_gender = Dense(32, activation='relu')(in_layer_gender)

feature = concatenate([dr_steps, feature_gender], axis=1)



o = Dense(1000, activation='relu')(feature)

o = Dense(1000, activation='relu')(o)

o = Dense(1, activation='linear')(o)



model = Model(inputs=[in_layer_img, in_layer_gender], outputs=o)



model.compile(loss='mean_absolute_error', optimizer=optim, metrics=[mae_months])



model.summary()
model_history = model.fit_generator(generator=train_flow,

    steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_flow,

    validation_steps=STEP_SIZE_VALID, epochs=EPOCHS,

    callbacks = callbacks_list)
loss_history = model_history.history['loss']

from keras.utils import plot_model

plot_model(model, to_file='model.png')

history_df = pd.DataFrame.from_dict(model_history.history)

history_df.to_csv('loss_history.csv')
test_X, test_Y = next(test_flow)
plt.style.use("dark_background")

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

pred_Y = boneage_div*model.predict(test_X, batch_size = 263, verbose = True)+boneage_mean

test_Y_months = boneage_div*test_Y+boneage_mean



ord_idx = np.argsort(test_Y)

ord_idx = ord_idx[np.linspace(0, len(ord_idx)-1, 4).astype(int)] # take 8 evenly spaced ones

fig, m_axs = plt.subplots(2, 2, figsize = (8, 8))

for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):

    cur_img = test_X[0][idx:(idx+1)]

    c_ax.imshow(cur_img[0, :,:,0], cmap = 'bone')

    

    c_ax.set_title('Age: %2.1fY\nPredicted Age: %2.1fY' % (test_Y_months[idx]/12.0, 

                                                           pred_Y[idx]/12.0))

    c_ax.axis('off')

fig.savefig('trained_img_predictions.png', dpi = 300)
from sklearn.metrics import mean_absolute_error as mean_abs

test_error = mean_abs(test_Y, pred_Y)

test_error = boneage_div * test_error + boneage_mean

test_error = str(test_error)



Outfile=open('test_error.txt','w')