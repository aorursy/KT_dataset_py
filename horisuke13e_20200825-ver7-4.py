### ver7.4



### DO

# 画像前処理: 平均画像を引く

# 精度確認: Holdoutを利用

# Transfer Learning: VGG16とVGG19のfine-tuning

# Data Augmentation: kerasのdatagen, random erasion, HSVのchannel dropout(もどき), TTA

# functional api: MLP*1 と CNN*4 の学習済みモデルを一つのMLPにまとめたMultimodal Modelを構築

# Ensemble: Random Seed Average, Model Ensemble(VGG16+VGG19), TTA Ensemble



### NOT DO

# 画像結合

# CrossValidation



### TO DO

# Pseudo Labeling

# GAN→半教師あり学習

# YOLOで物体検出
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

import glob

import cv2

import os

import random

%matplotlib inline



import category_encoders as ce

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import quantile_transform



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from keras.layers.advanced_activations import PReLU

from tensorflow.keras.applications import VGG16, VGG19, Xception

from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet50 import ResNet50

from keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator



DIR = '/kaggle/input/4th-datarobot-ai-academy-deep-learning'

DIR_IMG = '/kaggle/input/4th-datarobot-ai-academy-deep-learning/images'



ROOM_TYPE_LIST = ['bathroom', 'bedroom', 'frontal', 'kitchen']
# 計算に時間がかかる処理などは辞書形式でグローバル変数にキャッシュとして保存しておく

caches = {}
# 関数を実行した際にログを出力するデコレータ

def logger(func):

    def print_log(*args, **kwargs):

        print(f'{func.__name__}()')

        return func(*args, **kwargs)

    return print_log



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)
@logger

def get_df_concat(cache_name=None):

    global caches

    

    # キャッシュがあればキャッシュを返す

    if cache_name in caches.keys(): 

        print(' - already done, loading cache.')

        return caches[cache_name].copy()

    

    # キャッシュがなければ普通に読み込み

    df_train = pd.read_csv(f'{DIR}/train.csv')

    df_test = pd.read_csv(f'{DIR}/test.csv')

    df_train['is_test'] = 0

    df_test['is_test'] = 1

    df_concat = pd.concat([df_train, df_test], axis=0)

    

    # cache_nameが指定されている場合はキャッシュを残す

    if cache_name is not None: 

        caches[cache_name] = df_concat

        

    return df_concat.copy()



@logger

def get_images(df, size, cache_name=None):

    global caches

    if cache_name is not None:

        cache_name = f'{cache_name}_size{size}'

    

    # キャッシュがあればキャッシュを返す

    if cache_name in caches.keys(): 

        print(' - already done, loading cache.')

        return caches[cache_name][0].copy(), caches[cache_name][1].copy()

    

    # キャッシュがなければ普通に読み込み

    imgs_dict_train = {}

    imgs_dict_test = {}

    df_train = df_concat[df_concat['is_test'] != 1]

    df_test = df_concat[df_concat['is_test'] == 1]

    

    # train   

    for rtype in ROOM_TYPE_LIST:

        paths_train = df_train['id'].apply(lambda x: f'{DIR_IMG}/train_images/{x}_{rtype}.jpg').values

        images = [None]*999999

        count = 0

        for path in paths_train:

            image = cv2.imread(path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

            images[count] = image

            count += 1

        imgs_dict_train[rtype] = np.array(images[:count]) / 255.0

        

    # test 

    for rtype in ROOM_TYPE_LIST:

        paths_test = df_test['id'].apply(lambda x: f'{DIR_IMG}/test_images/{x}_{rtype}.jpg').values

        images = [None]*999999

        count = 0

        for path in paths_test:

            image = cv2.imread(path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

            images[count] = image

            count += 1

        imgs_dict_test[rtype] = np.array(images[:count]) / 255.0

        

    # cache_nameが指定されている場合はキャッシュを残す

    if cache_name is not None: 

        caches[cache_name] = (imgs_dict_train, imgs_dict_test)

        

    return imgs_dict_train.copy(), imgs_dict_test.copy()
@logger

def simple_feature_engineering(df):

    df['zipcode'] = df['zipcode'].astype(str)

    df['zipcode_1st_digit'] = df['zipcode'].str[:1]

    df['zipcode_1st2nd_digit'] = df['zipcode'].str[:2]

    df['zipcode_1st2nd3rd_digit'] = df['zipcode'].str[:3]

    df['area'] = df['area'].astype(str)

    df['area_1st_digit'] = df['area'].str[:1]

    df['area_1st2nd_digit'] = df['area'].str[:2]

    df['area_1st2nd3rd_digit'] = df['area'].str[:3]

    return df
@logger

def vgg16_fine_tuning(input_shape, hidden_layer_units=[128, 16, 8], layer_names_trainable=[], plot_summary=False, dropout=None):

    backbone = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    model = Sequential(layers=backbone.layers)

    model.add(Flatten())

    

    for n in hidden_layer_units:

        model.add(Dense(units=n, activation='relu', kernel_initializer='he_normal')) 

        if dropout is not None:

            model.add(Dropout(dropout))

    model.add(Dense(units=1, activation='linear'))

    

    model.trainable = True

    for i, layer in enumerate(model.layers):

        if i < len(backbone.layers) and layer.name not in layer_names_trainable:

            layer.trainable = False

            

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    if plot_summary:

        model.summary()

    return model
@logger

def vgg19_fine_tuning(input_shape, hidden_layer_units=[128, 16, 8], layer_names_trainable=[], plot_summary=False, dropout=None):

    backbone = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    model = Sequential(layers=backbone.layers)

    model.add(Flatten())

    

    for n in hidden_layer_units:

        model.add(Dense(units=n, activation='relu', kernel_initializer='he_normal')) 

        if dropout is not None:

            model.add(Dropout(dropout))

    model.add(Dense(units=1, activation='linear'))

    

    model.trainable = True

    for i, layer in enumerate(model.layers):

        if i < len(backbone.layers) and layer.name not in layer_names_trainable:

            layer.trainable = False

            

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    if plot_summary:

        model.summary()

    return model
@logger

def xception_fine_tuning(input_shape, hidden_layer_units=[128, 16, 8], layer_names_trainable=[], plot_summary=False, dropout=None):

    input_tensor = Input(shape=input_shape)

    xception_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_tensor=input_tensor)

    

    x = xception_model.output

    for n in hidden_layer_units:

        x = Dense(units=n, activation='relu', kernel_initializer='he_normal')(x)

        if dropout is not None:

            x = Dropout(dropout)(x)

    predictions = Dense(units=1, activation='linear')(x)

    model = Model(inputs=xception_model.input, outputs=predictions)

    

    model.trainable = True

    for i, layer in enumerate(model.layers):

        if i < len(xception_model.layers) and layer.name not in layer_names_trainable:

            layer.trainable = False

            

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    if plot_summary:

        model.summary()

    return model
@logger

def inception_v3_fine_tuning(input_shape, hidden_layer_units=[128, 16, 8], layer_names_trainable=[], plot_summary=False, dropout=None):

    input_tensor = Input(shape=input_shape)

    inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_tensor=input_tensor)

    

    x = inception_model.output

    for n in hidden_layer_units:

        x = Dense(units=n, activation='relu', kernel_initializer='he_normal')(x)

        if dropout is not None:

            x = Dropout(dropout)(x)

    predictions = Dense(units=1, activation='linear')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    

    model.trainable = True

    for i, layer in enumerate(model.layers):

        if i < len(inception_model.layers) and layer.name not in layer_names_trainable:

            layer.trainable = False

            

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    if plot_summary:

        model.summary()

    return model
@logger

def resnet50_fine_tuning(input_shape, hidden_layer_units=[128, 16, 8], layer_names_trainable=[], plot_summary=False, dropout=None):

    input_tensor = Input(shape=input_shape)

    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_tensor=input_tensor)

    

    x = resnet_model.output

    for n in hidden_layer_units:

        x = Dense(units=n, activation='relu', kernel_initializer='he_normal')(x)

        if dropout is not None:

            x = Dropout(dropout)(x)

    predictions = Dense(units=1, activation='linear')(x)

    model = Model(inputs=resnet_model.input, outputs=predictions)

    

    model.trainable = True

    for i, layer in enumerate(model.layers):

        if i < len(resnet_model.layers) and layer.name not in layer_names_trainable:

            layer.trainable = False

            

    model.compile(loss='mape', optimizer='adam', metrics=['mape'])

    if plot_summary:

        model.summary()

    return model
@logger

def mlp(input_shape, hidden_layer_units=[128, 16, 8], dropout=None):

    model = Sequential()

    

    for i, n in enumerate(hidden_layer_units):

        if i == 0:

            model.add(Dense(units=n, activation='relu', kernel_initializer='he_normal', input_shape=input_shape))

        else: 

            model.add(Dense(units=n, activation='relu', kernel_initializer='he_normal'))

        if dropout is not None:

            model.add(Dropout(dropout))

    model.add(Dense(units=1, activation='linear'))

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    

    return model
@logger

def mutimodal_model(input_shape_tbl, input_shape_img, trained_mlp_model, trained_cnn_models, hidden_layer_units_mlp_whole=[128, 16, 8], dropout_mlp=None, dropout_whole=None):

    

    # MLP on tabular

    input_tbl = Input(shape=input_shape_tbl)

    model_mlp_tbl = Sequential(layers=trained_mlp_model.layers)

    mlp_tbl_output = model_mlp_tbl(input_tbl)

    

    # CNNs on images

    arr_input_img, arr_cnn_output = [None]*len(trained_cnn_models), [None]*len(trained_cnn_models)

    for i, model_cnn in enumerate(trained_cnn_models):

        arr_input_img[i] = Input(shape=input_shape_img)

        model_cnn = Sequential(layers=model_cnn.layers)

        arr_cnn_output[i] = model_cnn(arr_input_img[i])

    

    # merge

    merged = tf.keras.layers.concatenate([mlp_tbl_output]+arr_cnn_output)

    x = merged

    for n in hidden_layer_units_mlp_whole:

        x = Dense(units=n, activation='relu', kernel_initializer='he_normal')(x)

        if dropout_whole is not None:

            x = Dropout(dropout_whole)(x)

    output = Dense(units=1, activation='linear')(x)

    model_mlp_whole = Model(inputs=[input_tbl]+arr_input_img, outputs=output)

    

    # freeze trained model

    model_mlp_whole.get_layer(mlp_tbl_output.name.split('/')[0]).trainable = False

    for cnn_output in arr_cnn_output:

        model_mlp_whole.get_layer(cnn_output.name.split('/')[0]).trainable = False

    

    # compile

    model_mlp_whole.compile(loss='mape', optimizer='adam', metrics=['mape'])

    

    return model_mlp_whole
# random erasion

def random_erasion(images, p=0.5, s=(0.02, 0.4), r=(0.3, 3)):

    result_images = [None]*len(images)

    for i, image in enumerate(images):

        result_image = np.copy(image)

        

        if np.random.rand() > p:

            result_images[i] = result_image

            

        else:

            # マスクする画素値をランダムで決める

            h, w, _ = result_image.shape

            mask_value = np.random.rand()

            mask_area = np.random.randint(h * w * s[0], h * w * s[1])

            mask_aspect_ratio = np.random.rand() * r[1] + r[0]



            # マスクのサイズとアスペクト比からマスクの高さと幅を決める

            mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))

            if mask_height > h - 1:

                mask_height = h - 1

            mask_width = int(mask_aspect_ratio * mask_height)

            if mask_width > w - 1:

                mask_width = w - 1



            top = np.random.randint(0, h - mask_height)

            left = np.random.randint(0, w - mask_width)

            bottom = top + mask_height

            right = left + mask_width

            result_image[top:bottom, left:right, :].fill(mask_value)



            result_images[i] = result_image

        

    return result_images
# random channel dropout (HSV)

def random_channel_dropout(images, p=0.5):

    result_images = [None]*len(images)

    for i, image in enumerate(images):

        if np.random.rand() > p:

            result_images[i] = np.copy(image)

        else:

            hsv_image = cv2.cvtColor(np.array(image*255, dtype=np.uint8), cv2.COLOR_RGB2HSV)

            hsv_image[:, :, np.random.randint(0, 2+1)] = np.random.randint(0, 255+1)

            rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)/255.0

            result_images[i] = rgb_image

    return result_images
def get_augmented_images(imgs_dict, x, y, datagen, n_aug=10, p_random_erasion=0.5, p_channel_dropout=0.5, shuffle=True):

    

    n_images = len(imgs_dict[ROOM_TYPE_LIST[0]])

    

    # no augmentation

    if n_aug == 0:

        return imgs_dict, np.array(x), np.array(y)

    

    # augmentation

    imgs_dict_aug = {rtype: [None]*n_images*n_aug for rtype in ROOM_TYPE_LIST}

    for rtype in ROOM_TYPE_LIST:

        for i, image_ori in enumerate(imgs_dict[rtype]):

            for j, image_aug in enumerate(datagen.flow(image_ori[np.newaxis], batch_size=1)):

                if j == n_aug:

                    break

                imgs_dict_aug[rtype][i*n_aug+j] = image_aug.reshape(IMG_SIZE, IMG_SIZE, 3)

    x_aug = sum([[record]*n_aug for record in x], [])

    y_aug = sum([[label]*n_aug for label in y], [])

    

    # random erasion

    for rtype in ROOM_TYPE_LIST:

        imgs_dict_aug[rtype] = random_erasion(imgs_dict_aug[rtype], p=p_random_erasion)

        

    # random channel dropout

    for rtype in ROOM_TYPE_LIST:

        imgs_dict_aug[rtype] = random_channel_dropout(imgs_dict_aug[rtype], p=p_channel_dropout)

        

    # shuffle

    if shuffle:

        zipped = list(zip(

            imgs_dict_aug[ROOM_TYPE_LIST[0]], imgs_dict_aug[ROOM_TYPE_LIST[1]], 

            imgs_dict_aug[ROOM_TYPE_LIST[2]], imgs_dict_aug[ROOM_TYPE_LIST[3]], 

            x_aug, y_aug

        ))

        np.random.shuffle(zipped)

        imgs_dict_aug[ROOM_TYPE_LIST[0]], imgs_dict_aug[ROOM_TYPE_LIST[1]], imgs_dict_aug[ROOM_TYPE_LIST[2]], imgs_dict_aug[ROOM_TYPE_LIST[3]], x_aug, y_aug = zip(*zipped)

        imgs_dict_aug = {rtype: np.asarray(imgs_dict_aug[rtype]) for rtype in ROOM_TYPE_LIST}

        

    return imgs_dict_aug, np.asarray(x_aug), np.asarray(y_aug)
def get_augmented_images_test(imgs_dict, x, datagen, p_random_erasion=0.5, p_channel_dropout=0.5):

    n_images = len(imgs_dict[ROOM_TYPE_LIST[0]])

    

    # augmentation

    imgs_dict_aug = {rtype: [None]*n_images for rtype in ROOM_TYPE_LIST}

    for rtype in ROOM_TYPE_LIST:

        for i, image_ori in enumerate(imgs_dict[rtype]):

            for j, image_aug in enumerate(datagen.flow(image_ori[np.newaxis], batch_size=1)):

                if j == 1:

                    break

                imgs_dict_aug[rtype][i] = image_aug.reshape(IMG_SIZE, IMG_SIZE, 3)

    x_aug = x

    

    # random erasion

    for rtype in ROOM_TYPE_LIST:

        imgs_dict_aug[rtype] = random_erasion(imgs_dict_aug[rtype], p=p_random_erasion)

        

    # random channel dropout

    for rtype in ROOM_TYPE_LIST:

        imgs_dict_aug[rtype] = random_channel_dropout(imgs_dict_aug[rtype], p=p_channel_dropout)

        

    imgs_dict_aug = {rtype: np.asarray(imgs_dict_aug[rtype]) for rtype in ROOM_TYPE_LIST}

        

    return imgs_dict_aug, np.asarray(x_aug)
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def my_train_test_split(df_train, imgs_dict_train, valid_rate=0.2, shuffle=True):

    df_train = df_train.reset_index(drop=True).copy()

    

    ix_valid = random.sample(df_train.index.tolist(), int(len(df_train)*valid_rate))

    ix_train = [i for i in df_train.index.tolist() if i not in ix_valid]

    

    imgs_dict_valid = {}

    for rtype in ROOM_TYPE_LIST:

        images = imgs_dict_train[rtype].copy()

        imgs_dict_train[rtype] = images[ix_train]

        imgs_dict_valid[rtype] = images[ix_valid]

        

    return df_train.iloc[ix_train], df_train.iloc[ix_valid], imgs_dict_train, imgs_dict_valid
HOLDOUT_FLG = False

RATE_HOLDOUT = 0.2

RATE_VALID = 0.2



IMG_SIZE = 64

N_AUG = 4

P_RANDOM_ERASION = 0.25

P_CHANNEL_DROPOUT = 0.25



dropout_mlp = 0.1

dropout_cnn = 0.5

dropout_whole = None



hidden_layer_units_mlp_tbl = [128, 16, 8]

hidden_layer_units_cnn = [256, 32]

hidden_layer_units_mlp_whole = [16, 8]

layer_names_trainable_vgg16 = ['block5_conv1', 'block5_conv2', 'block5_conv3']

layer_names_trainable_vgg19 = ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4']



datagen = ImageDataGenerator(

    horizontal_flip=True, vertical_flip=False, 

    zoom_range=0.2, rotation_range=15,

    width_shift_range=0.2, height_shift_range=0.2

)



n_seeds = 5
es = EarlyStopping(patience=5, mode='min', verbose=0)

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=0,  mode='min')





### 1. Import Data



df_concat = get_df_concat(cache_name=None)

imgs_dict_train_ori, imgs_dict_test_ori = get_images(df_concat, IMG_SIZE, cache_name=None)



# Subtract Mean Image

for rtype in ROOM_TYPE_LIST:

    images = np.vstack([imgs_dict_train_ori[rtype], imgs_dict_test_ori[rtype]])

    imgs_dict_train_ori[rtype] -= np.mean(images, axis=0)

    imgs_dict_test_ori[rtype] -= np.mean(images, axis=0)



    

### 2. Feature Engineering



df_concat = simple_feature_engineering(df_concat)



col_target = 'price'

cols_unnecessary = ['id', 'is_test']

cols_quantitive = [col for col in df_concat.columns if (df_concat[col].dtype != 'object') and (col != col_target) and (col not in cols_unnecessary)]

cols_qualitative = [col for col in df_concat.columns if (df_concat[col].dtype == 'object') and (col != col_target) and (col not in cols_unnecessary)]





### 3. Encoding, Transformation



# One-Hot Encoding

ohe = ce.OneHotEncoder(cols=cols_qualitative, handle_unknown='ignore', use_cat_names=True)

df_ohe = ohe.fit_transform(df_concat[cols_qualitative])

df_concat = pd.concat([df_concat, df_ohe], axis=1)

df_concat = df_concat.drop(columns=cols_qualitative)



# rank-gauss

df_concat[cols_quantitive] = quantile_transform(df_concat[cols_quantitive], n_quantiles=100, random_state=0, output_distribution='normal')





### 4. Split Data



cols_to_use = [col for col in df_concat.columns if (col != col_target) and (col not in cols_unnecessary)]



# train/test

df_train_ori = df_concat[df_concat['is_test'] != 1].reset_index(drop=True).copy()

df_test_ori = df_concat[df_concat['is_test'] == 1].reset_index(drop=True).copy()

train_y_ori = df_train_ori['price'].values

train_x_ori = np.asarray(df_train_ori[cols_to_use].values).astype(np.float32)

test_x_ori = np.asarray(df_test_ori[cols_to_use].values).astype(np.float32)





### 5. Modeling



y_preds_tta = []

y_preds_not_tta = []

valid_scores, holdout_scores = [], []

train_images_aug_sample = None



for seed in range(n_seeds):

    print(f'\n### seed {seed}\n')

    seed_everything(seed)

    

    

    ### 4. Re-Split Data

    

    df_train, df_test = df_train_ori.copy(), df_test_ori.copy()

    train_x, train_y = train_x_ori.copy(), train_y_ori.copy()

    test_x = test_x_ori.copy()

    imgs_dict_train, imgs_dict_test = imgs_dict_train_ori.copy(), imgs_dict_test_ori.copy()



    # holdout

    df_holdout = None

    imgs_dict_holdout = None

    holdout_y, holdout_x = None, None

    if HOLDOUT_FLG:

        df_train, df_holdout, imgs_dict_train, imgs_dict_holdout = my_train_test_split(df_train, imgs_dict_train, valid_rate=RATE_HOLDOUT)

        train_x, train_y = np.asarray(df_train[cols_to_use].values).astype(np.float32), df_train['price'].values

        holdout_x, holdout_y = np.asarray(df_holdout[cols_to_use].values).astype(np.float32), df_holdout['price'].values



    # validation

    df_train, df_valid, imgs_dict_train, imgs_dict_valid = my_train_test_split(df_train, imgs_dict_train, valid_rate=RATE_VALID)

    train_x, train_y = np.asarray(df_train[cols_to_use].values).astype(np.float32), df_train['price'].values

    valid_x, valid_y = np.asarray(df_valid[cols_to_use].values).astype(np.float32), df_valid['price'].values

    

    

    ### 5.1. Modeling (1) MLP on Tabular

    

    input_shape_tbl = (train_x.shape[1], )



    # train model

    filepath = 'mlp_tabular.hdf5'

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

    model_mlp_tbl = mlp(input_shape=input_shape_tbl, hidden_layer_units=hidden_layer_units_mlp_tbl, dropout=dropout_mlp)

    history = model_mlp_tbl.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=1000, batch_size=32, callbacks=[es, checkpoint, reduce_lr_loss], verbose=0)



    # evaluation

    model_mlp_tbl.load_weights(filepath)

    valid_pred = model_mlp_tbl.predict(valid_x, batch_size=32)[:, 0]

    print(f'### Evaluation: {filepath}')

    print(f'# MAPE (validation): {mean_absolute_percentage_error(valid_y, valid_pred):.2f}')

    if HOLDOUT_FLG:

        holdout_pred = model_mlp_tbl.predict(holdout_x, batch_size=32)[:, 0]

        print(f'# MAPE (holdout)   : {mean_absolute_percentage_error(holdout_y, holdout_pred):.2f}')

    

    

    ### 5.2. Modeling (2) CNNs on Images

    

    input_shape_img = (IMG_SIZE, IMG_SIZE, 3)

    

    # augmentation

    imgs_dict_train_aug, train_x_aug, train_y_aug = get_augmented_images(imgs_dict_train, train_x, train_y, datagen, n_aug=N_AUG, p_random_erasion=P_RANDOM_ERASION, p_channel_dropout=P_CHANNEL_DROPOUT, shuffle=True)

    imgs_dict_valid_aug, valid_x_aug, valid_y_aug = get_augmented_images(imgs_dict_valid, valid_x, valid_y, datagen, n_aug=N_AUG, p_random_erasion=P_RANDOM_ERASION, p_channel_dropout=P_CHANNEL_DROPOUT, shuffle=True)

    imgs_dict_holdout_aug, holdout_x_aug, holdout_y_aug = get_augmented_images(imgs_dict_holdout, holdout_x, holdout_y, datagen, n_aug=N_AUG, p_random_erasion=P_RANDOM_ERASION, p_channel_dropout=P_CHANNEL_DROPOUT, shuffle=True) if HOLDOUT_FLG else (None, None, None)

    train_images_aug_sample = imgs_dict_train_aug[ROOM_TYPE_LIST[0]][:20]

    

    # train models

    cnn_models = [None]*len(ROOM_TYPE_LIST)

    for i, rtype in enumerate(ROOM_TYPE_LIST):

        filepath = f'vgg16_ft_{rtype}.hdf5'

        checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto')

        cnn_models[i] = vgg16_fine_tuning(input_shape=input_shape_img, hidden_layer_units=hidden_layer_units_cnn, layer_names_trainable=layer_names_trainable_vgg16, dropout=dropout_cnn)

        history = cnn_models[i].fit(imgs_dict_train_aug[rtype], train_y_aug, validation_data=(imgs_dict_valid_aug[rtype], valid_y_aug), epochs=100, batch_size=32, callbacks=[es, checkpoint, reduce_lr_loss], verbose=0)



        # evaluation

        cnn_models[i].load_weights(filepath)

        valid_pred_aug = cnn_models[i].predict(imgs_dict_valid_aug[rtype], batch_size=32)[:, 0]

        print(f'### Evaluation: {filepath}')

        print(f'# MAPE (validation): {mean_absolute_percentage_error(valid_y_aug, valid_pred_aug):.2f}')

        if HOLDOUT_FLG:

            holdout_pred_aug = cnn_models[i].predict(imgs_dict_holdout_aug[rtype], batch_size=32)[:, 0]

            print(f'# MAPE (holdout)   : {mean_absolute_percentage_error(holdout_y_aug, holdout_pred_aug):.2f}')

        

    

    ### 5.3. Modeling (3) Mutimodal Model on Tabular and Images

    

    train_x_mm = (train_x, imgs_dict_train[ROOM_TYPE_LIST[0]], imgs_dict_train[ROOM_TYPE_LIST[1]], imgs_dict_train[ROOM_TYPE_LIST[2]], imgs_dict_train[ROOM_TYPE_LIST[3]])

    valid_x_mm = (valid_x, imgs_dict_valid[ROOM_TYPE_LIST[0]], imgs_dict_valid[ROOM_TYPE_LIST[1]], imgs_dict_valid[ROOM_TYPE_LIST[2]], imgs_dict_valid[ROOM_TYPE_LIST[3]])

    holdout_x_mm = (holdout_x, imgs_dict_holdout[ROOM_TYPE_LIST[0]], imgs_dict_holdout[ROOM_TYPE_LIST[1]], imgs_dict_holdout[ROOM_TYPE_LIST[2]], imgs_dict_holdout[ROOM_TYPE_LIST[3]])  if HOLDOUT_FLG else (None, None, None, None, None)

    

    # train model

    filepath = f'multimodal_model.hdf5'

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto')

    model_whole = mutimodal_model(

        input_shape_tbl=input_shape_tbl, input_shape_img=input_shape_img, 

        trained_mlp_model=model_mlp_tbl, trained_cnn_models=cnn_models, 

        hidden_layer_units_mlp_whole=hidden_layer_units_mlp_whole, 

        dropout_mlp=dropout_mlp, dropout_whole=dropout_whole

    )

    history = model_whole.fit(

        train_x_mm, train_y, validation_data=(valid_x_mm, valid_y), 

        epochs=100, batch_size=32, callbacks=[es, checkpoint, reduce_lr_loss], verbose=0

    )



    # evaluation

    model_whole.load_weights(filepath)

    valid_pred = model_whole.predict(valid_x_mm, batch_size=32)[:, 0]

    valid_score = mean_absolute_percentage_error(valid_y, valid_pred)

    valid_scores.append(valid_score)

    print(f'### Evaluation: {filepath}')

    print(f'# MAPE (validation): {valid_score:.2f}')

    if HOLDOUT_FLG:

        holdout_pred = model_whole.predict(holdout_x_mm, batch_size=32)[:, 0]

        holdout_score = mean_absolute_percentage_error(holdout_y, holdout_pred)

        holdout_scores.append(holdout_score)

        print(f'# MAPE (holdout)   : {holdout_score:.2f}')

        

    # plot model summary

    if seed == n_seeds-1:

        print('\n### Multimodal Model Summary\n')

        print(pd.DataFrame({

            'layer_name': [l.name for l in model_whole.layers], 

            'trainable': [l.trainable for l in model_whole.layers]

        }).set_index('layer_name'))

        model_whole.summary()

        plot_model(model_whole)

    

    

    ### 6. Prediction

    

    # Normal Prediction

    test_x_mm = (test_x, imgs_dict_test[ROOM_TYPE_LIST[0]], imgs_dict_test[ROOM_TYPE_LIST[1]], imgs_dict_test[ROOM_TYPE_LIST[2]], imgs_dict_test[ROOM_TYPE_LIST[3]])

    y_pred = model_whole.predict(test_x_mm, batch_size=32)[:, 0]

    y_preds_not_tta.append(y_pred)

    

    # Avg of Test Time Augmentation

    y_preds_tta_tmp = []

    for i in range(N_AUG):

        imgs_dict_test_aug, _ = get_augmented_images_test(imgs_dict_test, test_x, datagen, p_random_erasion=P_RANDOM_ERASION, p_channel_dropout=P_CHANNEL_DROPOUT)

        test_x_mm = (test_x, imgs_dict_test_aug[ROOM_TYPE_LIST[0]], imgs_dict_test_aug[ROOM_TYPE_LIST[1]], imgs_dict_test_aug[ROOM_TYPE_LIST[2]], imgs_dict_test_aug[ROOM_TYPE_LIST[3]])

        y_pred = model_whole.predict(test_x_mm, batch_size=32)[:, 0]

        y_preds_tta_tmp.append(y_pred)

    y_preds_tta.append(np.mean(np.array(y_preds_tta_tmp), axis=0))
### 5. Modeling



for seed in range(n_seeds):

    seed += n_seeds

    

    print(f'\n### seed {seed}\n')

    seed_everything(seed)

    

    

    ### 4. Re-Split Data

    

    df_train, df_test = df_train_ori.copy(), df_test_ori.copy()

    train_x, train_y = train_x_ori.copy(), train_y_ori.copy()

    test_x = test_x_ori.copy()

    imgs_dict_train, imgs_dict_test = imgs_dict_train_ori.copy(), imgs_dict_test_ori.copy()



    # holdout

    df_holdout = None

    imgs_dict_holdout = None

    holdout_y, holdout_x = None, None

    if HOLDOUT_FLG:

        df_train, df_holdout, imgs_dict_train, imgs_dict_holdout = my_train_test_split(df_train, imgs_dict_train, valid_rate=RATE_HOLDOUT)

        train_x, train_y = np.asarray(df_train[cols_to_use].values).astype(np.float32), df_train['price'].values

        holdout_x, holdout_y = np.asarray(df_holdout[cols_to_use].values).astype(np.float32), df_holdout['price'].values



    # validation

    df_train, df_valid, imgs_dict_train, imgs_dict_valid = my_train_test_split(df_train, imgs_dict_train, valid_rate=RATE_VALID)

    train_x, train_y = np.asarray(df_train[cols_to_use].values).astype(np.float32), df_train['price'].values

    valid_x, valid_y = np.asarray(df_valid[cols_to_use].values).astype(np.float32), df_valid['price'].values

    

    

    ### 5.1. Modeling (1) MLP on Tabular

    

    input_shape_tbl = (train_x.shape[1], )



    # train model

    filepath = 'mlp_tabular.hdf5'

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

    model_mlp_tbl = mlp(input_shape=input_shape_tbl, hidden_layer_units=hidden_layer_units_mlp_tbl, dropout=dropout_mlp)

    history = model_mlp_tbl.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=1000, batch_size=32, callbacks=[es, checkpoint, reduce_lr_loss], verbose=0)



    # evaluation

    model_mlp_tbl.load_weights(filepath)

    valid_pred = model_mlp_tbl.predict(valid_x, batch_size=32)[:, 0]

    print(f'### Evaluation: {filepath}')

    print(f'# MAPE (validation): {mean_absolute_percentage_error(valid_y, valid_pred):.2f}')

    if HOLDOUT_FLG:

        holdout_pred = model_mlp_tbl.predict(holdout_x, batch_size=32)[:, 0]

        print(f'# MAPE (holdout)   : {mean_absolute_percentage_error(holdout_y, holdout_pred):.2f}')

    

    

    ### 5.2. Modeling (2) CNNs on Images

    

    input_shape_img = (IMG_SIZE, IMG_SIZE, 3)

    

    # augmentation

    imgs_dict_train_aug, train_x_aug, train_y_aug = get_augmented_images(imgs_dict_train, train_x, train_y, datagen, n_aug=N_AUG, p_random_erasion=P_RANDOM_ERASION, p_channel_dropout=P_CHANNEL_DROPOUT, shuffle=True)

    imgs_dict_valid_aug, valid_x_aug, valid_y_aug = get_augmented_images(imgs_dict_valid, valid_x, valid_y, datagen, n_aug=N_AUG, p_random_erasion=P_RANDOM_ERASION, p_channel_dropout=P_CHANNEL_DROPOUT, shuffle=True)

    imgs_dict_holdout_aug, holdout_x_aug, holdout_y_aug = get_augmented_images(imgs_dict_holdout, holdout_x, holdout_y, datagen, n_aug=N_AUG, p_random_erasion=P_RANDOM_ERASION, p_channel_dropout=P_CHANNEL_DROPOUT, shuffle=True) if HOLDOUT_FLG else (None, None, None)

    

    # train models

    cnn_models = [None]*len(ROOM_TYPE_LIST)

    for i, rtype in enumerate(ROOM_TYPE_LIST):

        filepath = f'vgg19_ft_{rtype}.hdf5'

        checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto')

        cnn_models[i] = vgg19_fine_tuning(input_shape=input_shape_img, hidden_layer_units=hidden_layer_units_cnn, layer_names_trainable=layer_names_trainable_vgg19, dropout=dropout_cnn)

        history = cnn_models[i].fit(imgs_dict_train_aug[rtype], train_y_aug, validation_data=(imgs_dict_valid_aug[rtype], valid_y_aug), epochs=100, batch_size=32, callbacks=[es, checkpoint, reduce_lr_loss], verbose=0)



        # evaluation

        cnn_models[i].load_weights(filepath)

        valid_pred_aug = cnn_models[i].predict(imgs_dict_valid_aug[rtype], batch_size=32)[:, 0]

        print(f'### Evaluation: {filepath}')

        print(f'# MAPE (validation): {mean_absolute_percentage_error(valid_y_aug, valid_pred_aug):.2f}')

        if HOLDOUT_FLG:

            holdout_pred_aug = cnn_models[i].predict(imgs_dict_holdout_aug[rtype], batch_size=32)[:, 0]

            print(f'# MAPE (holdout)   : {mean_absolute_percentage_error(holdout_y_aug, holdout_pred_aug):.2f}')

        

    

    ### 5.3. Modeling (3) Mutimodal Model on Tabular and Images

    

    train_x_mm = (train_x, imgs_dict_train[ROOM_TYPE_LIST[0]], imgs_dict_train[ROOM_TYPE_LIST[1]], imgs_dict_train[ROOM_TYPE_LIST[2]], imgs_dict_train[ROOM_TYPE_LIST[3]])

    valid_x_mm = (valid_x, imgs_dict_valid[ROOM_TYPE_LIST[0]], imgs_dict_valid[ROOM_TYPE_LIST[1]], imgs_dict_valid[ROOM_TYPE_LIST[2]], imgs_dict_valid[ROOM_TYPE_LIST[3]])

    holdout_x_mm = (holdout_x, imgs_dict_holdout[ROOM_TYPE_LIST[0]], imgs_dict_holdout[ROOM_TYPE_LIST[1]], imgs_dict_holdout[ROOM_TYPE_LIST[2]], imgs_dict_holdout[ROOM_TYPE_LIST[3]])  if HOLDOUT_FLG else (None, None, None, None, None)

    

    # train model

    filepath = f'multimodal_model.hdf5'

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto')

    model_whole = mutimodal_model(

        input_shape_tbl=input_shape_tbl, input_shape_img=input_shape_img, 

        trained_mlp_model=model_mlp_tbl, trained_cnn_models=cnn_models, 

        hidden_layer_units_mlp_whole=hidden_layer_units_mlp_whole, 

        dropout_mlp=dropout_mlp, dropout_whole=dropout_whole

    )

    history = model_whole.fit(

        train_x_mm, train_y, validation_data=(valid_x_mm, valid_y), 

        epochs=100, batch_size=32, callbacks=[es, checkpoint, reduce_lr_loss], verbose=0

    )



    # evaluation

    model_whole.load_weights(filepath)

    valid_pred = model_whole.predict(valid_x_mm, batch_size=32)[:, 0]

    valid_score = mean_absolute_percentage_error(valid_y, valid_pred)

    valid_scores.append(valid_score)

    print(f'### Evaluation: {filepath}')

    print(f'# MAPE (validation): {valid_score:.2f}')

    if HOLDOUT_FLG:

        holdout_pred = model_whole.predict(holdout_x_mm, batch_size=32)[:, 0]

        holdout_score = mean_absolute_percentage_error(holdout_y, holdout_pred)

        holdout_scores.append(holdout_score)

        print(f'# MAPE (holdout)   : {holdout_score:.2f}')

    

    

    ### 6. Prediction

    

    # Normal Prediction

    test_x_mm = (test_x, imgs_dict_test[ROOM_TYPE_LIST[0]], imgs_dict_test[ROOM_TYPE_LIST[1]], imgs_dict_test[ROOM_TYPE_LIST[2]], imgs_dict_test[ROOM_TYPE_LIST[3]])

    y_pred = model_whole.predict(test_x_mm, batch_size=32)[:, 0]

    y_preds_not_tta.append(y_pred)

    

    # Avg of Test Time Augmentation

    y_preds_tta_tmp = []

    for i in range(N_AUG):

        imgs_dict_test_aug, _ = get_augmented_images_test(imgs_dict_test, test_x, datagen, p_random_erasion=P_RANDOM_ERASION, p_channel_dropout=P_CHANNEL_DROPOUT)

        test_x_mm = (test_x, imgs_dict_test_aug[ROOM_TYPE_LIST[0]], imgs_dict_test_aug[ROOM_TYPE_LIST[1]], imgs_dict_test_aug[ROOM_TYPE_LIST[2]], imgs_dict_test_aug[ROOM_TYPE_LIST[3]])

        y_pred = model_whole.predict(test_x_mm, batch_size=32)[:, 0]

        y_preds_tta_tmp.append(y_pred)

    y_preds_tta.append(np.mean(np.array(y_preds_tta_tmp), axis=0))
### 7. save submission (VGG16 VGG19 model ensemble + tta ensemble + random seed ensemble)



version = '7_4'

now = dt.datetime.now().strftime('%Y%m%d')



# ensemble all results (TTA+Normal)

df_submission = pd.read_csv(f'{DIR}/sample_submission.csv')

predictions = np.mean(np.array(

    [np.mean(np.array(y_preds_tta), axis=0), np.mean(np.array(y_preds_not_tta), axis=0)]

), axis=0)

df_submission['price'] = predictions

df_submission['price'] = df_submission['price'].astype(int)

f_name_out = f'{now}_submission_ver{version}_ens_all_preds_tta+normal.csv'

df_submission.to_csv(f_name_out, index=False)

# ensemble all results (Normal)

df_submission = pd.read_csv(f'{DIR}/sample_submission.csv')

df_submission['price'] = np.mean(np.array(y_preds_not_tta), axis=0)

df_submission['price'] = df_submission['price'].astype(int)

f_name_out = f'{now}_submission_ver{version}_ens_all_preds_normal.csv'

df_submission.to_csv(f_name_out, index=False)



# ensemble only vgg16 (TTA+Normal)

df_submission = pd.read_csv(f'{DIR}/sample_submission.csv')

predictions = np.mean(np.array(

    [np.mean(np.array(y_preds_tta[:n_seeds]), axis=0), np.mean(np.array(y_preds_not_tta[:n_seeds]), axis=0)]

), axis=0)

df_submission['price'] = predictions

df_submission['price'] = df_submission['price'].astype(int)

f_name_out = f'{now}_submission_ver{version}_ens_only_vgg16_preds_tta+normal.csv'

df_submission.to_csv(f_name_out, index=False)



# ensemble only vgg19 (TTA+Normal)

df_submission = pd.read_csv(f'{DIR}/sample_submission.csv')

predictions = np.mean(np.array(

    [np.mean(np.array(y_preds_tta[::-1][:n_seeds]), axis=0), np.mean(np.array(y_preds_not_tta[::-1][:n_seeds]), axis=0)]

), axis=0)

df_submission['price'] = predictions

df_submission['price'] = df_submission['price'].astype(int)

f_name_out = f'{now}_submission_ver{version}_ens_only_vgg19_preds_tta+normal.csv'

df_submission.to_csv(f_name_out, index=False)

# ensemble only vgg19 (Normal)

df_submission = pd.read_csv(f'{DIR}/sample_submission.csv')

df_submission['price'] = np.mean(np.array(y_preds_not_tta[::-1][:n_seeds]), axis=0)

df_submission['price'] = df_submission['price'].astype(int)

f_name_out = f'{now}_submission_ver{version}_ens_only_vgg19_preds_normal.csv'

df_submission.to_csv(f_name_out, index=False)



# ensemble only top5 validation score

sr_tmp = pd.Series(valid_scores).sort_values(ascending=False)

ix_preds_top5 = sr_tmp.index.tolist()[:5]

print(f'prediction index of top5 scores: {ix_preds_top5}')

# ensemble only top5 validation score (TTA+Normal)

df_submission = pd.read_csv(f'{DIR}/sample_submission.csv')

predictions = np.mean(np.array(

    [np.mean(np.array(y_preds_tta)[ix_preds_top5], axis=0), np.mean(np.array(y_preds_not_tta)[ix_preds_top5], axis=0)]

), axis=0)

df_submission['price'] = predictions

df_submission['price'] = df_submission['price'].astype(int)

f_name_out = f'{now}_submission_ver{version}_ens_only_top5_preds_tta+normal.csv'

df_submission.to_csv(f_name_out, index=False)

# ensemble only top5 validation score (Normal)

df_submission = pd.read_csv(f'{DIR}/sample_submission.csv')

df_submission['price'] = np.mean(np.array(y_preds_not_tta)[ix_preds_top5], axis=0)

df_submission['price'] = df_submission['price'].astype(int)

f_name_out = f'{now}_submission_ver{version}_ens_only_top5_preds_normal.csv'

df_submission.to_csv(f_name_out, index=False)



print(' > done')
print(f'Average of Validation Scores: {np.mean(valid_scores):.2f}')



df_result = pd.DataFrame({

    'cnn model': ['VGG16 (fine-tuning)']*n_seeds+['VGG19 (fine-tuning)']*n_seeds, 

    'seed': [seed for seed in range(n_seeds)]+[seed+n_seeds for seed in range(n_seeds)], 

    'validation score': valid_scores, 

    'holdout score': holdout_scores if HOLDOUT_FLG else [None]*n_seeds*2

})

df_result
for image in train_images_aug_sample:

    plt.imshow(image)

    plt.show() 