# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

from pandas import DataFrame, Series

import numpy as np

import random

import os

import datetime

import glob

import cv2

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import plot_model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Dropout, Activation

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm_notebook as tqdm



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



seed_everything(2020)    
# データの読み込み

df_train = pd.read_csv('/kaggle/input/4th-datarobot-ai-academy-deep-learning/train.csv')

df_test = pd.read_csv('/kaggle/input/4th-datarobot-ai-academy-deep-learning/test.csv')

df_train.head()
df_train.describe()
df_train.describe()
# 欠損値の確認

df_train.isnull().sum()
df_train.isnull().sum()
target = 'price'



plt.figure(figsize=[7,7])

df_train[target].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(target)

plt.ylabel('density')

plt.show()
fig=plt.figure(figsize=[20,60])



i = 0

for col in df_test.columns:

    i=i+1

    plt.subplots_adjust(wspace=0.2, hspace=0.8)

    ax_name = fig.add_subplot(40,2,i)

    ax_name.hist(df_train[col],bins=30,density=True, alpha=0.5,color = 'r')

    ax_name.hist(df_test[col],bins=30,density=True, alpha=0.5, color = 'b')

    ax_name.set_title(col)
fig=plt.figure(figsize=[20,60])



i = 0

for col in df_train.columns:

    i=i+1

    plt.subplots_adjust(wspace=0.2, hspace=0.8)

    ax_name = fig.add_subplot(10,2,i)

    ax_name.scatter(df_train[col], df_train['price'])

    ax_name.set_title(col)
"""

画像解析の際に、条件がいいはずなのに安い物件を除外すると高い精度が期待できるのかも

#id 41,         153, 157, 290, 389, 462, 532

id  41, 22, 37, 153,      290, 389, 532, 90, 134, 235, 257, 462

df_train.drop(df_train.index[df_train['area'] > 8000], inplace=True)# 41, 532 # 敷地が広いけど安い物件

df_train.drop(df_train.index[df_train['bedrooms'] >= 7], inplace=True) # 90, 134, 235, 257, 462 # 条件がいいけど安い

df_train.drop(df_train.index[df_train['bathrooms'] > 5], inplace=True) # 22, 37 # 条件がいいけど安い

df_train.drop(df_train.index[df_train['id'] == 153], inplace=True) # 低価格

df_train.drop(df_train.index[df_train['id'] == 290], inplace=True) # 低価格

df_train.drop(df_train.index[df_train['id'] == 389], inplace=True) # 低価格





"""
# area の外れ値

df_train[df_train['area'] > 8000]
df_train.drop(df_train.index[df_train['area'] > 8000], inplace=True)
# bedrooms が7以上

df_train[df_train['bedrooms'] >= 7]
df_train.drop(df_train.index[df_train['bedrooms'] >= 7], inplace=True)
# bathrooms が6以上

df_train[df_train['bathrooms'] > 5]
df_train.drop(df_train.index[df_train['bathrooms'] > 5], inplace=True)
fig=plt.figure(figsize=[20,60])



i = 0

for col in df_train.columns:

    i=i+1

    plt.subplots_adjust(wspace=0.2, hspace=0.8)

    ax_name = fig.add_subplot(10,2,i)

    ax_name.scatter(df_train[col], df_train['price'])

    ax_name.set_title(col)
# # area の外れ値

# df_train[df_train['area'] < 2000].sort_values('price')

#df_train.drop(df_train.index[df_train['id'] == 157], inplace=True)



# 狭いけど高めの家を削除するのをやめる
#相関係数の確認

df_corr = df_train.corr()

print(df_corr)

sns.heatmap(df_corr, vmax=1, vmin=-1, center=0, cmap = 'seismic')
for col in df_train.columns:

    print(col, df_train[col].dtype)
df_train['area'] = df_train['area'].apply(np.log1p)

df_train['price'] = df_train['price'].apply(np.log1p)



df_test['area'] = df_test['area'].apply(np.log1p)

cols = ['area']



scaler = StandardScaler()

scaler.fit(df_train[cols])



df_train[cols] = scaler.transform(df_train[cols])

df_test[cols] = scaler.transform(df_test[cols])
# scaler = StandardScaler()

# df_train[['price']] = scaler.fit_transform(df_train[['price']])



# #変換を元に戻す際は、以下を実行

# df_train[['price']] = scaler.inverse_transform(df_train[['price']])
# zipcodeはカテゴリデータであるが、値が大きすぎるため、線型モデルにも使えるようにスケールを小さくする。

df_train['zipcode'] = df_train['zipcode'] /10000

df_test['zipcode'] = df_test['zipcode'] /10000 
#df_train[df_train['price'] < 11]
df_train.drop(df_train.index[df_train['id'] == 153], inplace=True)

df_train.drop(df_train.index[df_train['id'] == 290], inplace=True)

df_train.drop(df_train.index[df_train['id'] == 389], inplace=True)

fig=plt.figure(figsize=[20,60])



i = 0

for col in df_train.columns:

    i=i+1

    plt.subplots_adjust(wspace=0.2, hspace=0.8)

    ax_name = fig.add_subplot(10,2,i)

    ax_name.scatter(df_train[col], df_train['price'])

    ax_name.set_title(col)
target = 'price'



plt.figure(figsize=[7,7])

df_train[target].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(target)

plt.ylabel('density')

plt.show()
# Xとyの分離

y_train = df_train['price']

X_train = df_train.drop(['price'], axis=1)

X_train = X_train.drop(['id'], axis=1)



X_test = df_test

X_test = X_test.drop(['id'], axis=1)
X_train.shape, X_test.shape
# # Target Encoding は実施しない方が、精度がよい

# target = 'price'

# X_temp = pd.concat([X_train, y_train], axis=1)



# te_cols = ['zipcode']





# #for col in cats: #te_cols:

# for col in te_cols:    



#     # X_testはX_trainでエンコーディングする

#     summary = X_temp.groupby([col])[target].mean()

#     X_test['te_' + col] = X_test[col].map(summary) 





#     # X_trainのカテゴリ変数をoofでエンコーディングする

#     skf = KFold(n_splits=5, random_state=71, shuffle=True)

#     enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



#     for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#         X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#         X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#         summary = X_train_.groupby([col])[target].mean()

#         enc_train.iloc[val_ix] = X_val[col].map(summary)

        

#     X_train['te_' + col]  = enc_train
fig=plt.figure(figsize=[20,60])



i = 0

for col in X_test.columns:

    i=i+1

    plt.subplots_adjust(wspace=0.2, hspace=0.8)

    ax_name = fig.add_subplot(40,2,i)

    ax_name.hist(X_train[col],bins=30,density=True, alpha=0.5,color = 'r')

    ax_name.hist(X_test[col],bins=30,density=True, alpha=0.5, color = 'b')

    ax_name.set_title(col)
X_train.isnull().sum()
X_test.isnull().sum()
# # target encodingによって発生した欠損値については、中央値で欠損値を補完

# X_train['te_bedrooms'] = X_train['te_bedrooms'].fillna(X_train['te_bedrooms'].median())

# X_train['te_bathrooms'] = X_train['te_bathrooms'].fillna(X_train['te_bathrooms'].median())

# X_train['te_zipcode'] = X_train['te_zipcode'].fillna(X_train['te_zipcode'].median())



# X_test['te_zipcode'] = X_test['te_zipcode'].fillna(X_test['te_zipcode'].median())
X_train.isnull().sum()
X_test.isnull().sum()
X_train = X_train.drop(['bedrooms'], axis=1)

X_test = X_test.drop(['bedrooms'], axis=1)
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# filepath = "mlp_best_model.hdf5"



# score = []

# y_pred_table_avg = np.zeros(len(X_test))

# y_pred_table = []



# n_folds = 5



# scores = []



# for random_state in range(n_folds):

#     print("Training on Fold: ",random_state+1)

#     X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.1, 

#                                                         random_state = random_state, shuffle=True)

#     seed_everything(random_state)



#     model = Sequential()

#     model.add(Dense(units=256, input_shape = (len(X_train.columns),), kernel_initializer='he_normal',activation='relu'))    

# #   model.add(Dropout(0.3))

#     model.add(Dense(units=64,  kernel_initializer='he_normal',activation='relu'))

# #   model.add(Dropout(0.3))

#     model.add(Dense(units=32, kernel_initializer='he_normal', activation='relu'))     

# #   model.add(Dropout(0.3))

#     model.add(Dense(1, activation='linear'))

#     model.compile(loss='mape', optimizer='adam', metrics=['mape']) 



#     es = EarlyStopping(patience=10, mode='min', verbose=1) 



#     checkpoint = ModelCheckpoint(monitor='val_loss',filepath=filepath, save_best_only=True, mode='auto') 



#     reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, mode='min', factor=0.1)



#     history = model.fit(X_train_, y_train_, batch_size=32, epochs=30, validation_data=(X_val, y_val), 

#                     callbacks=[es, checkpoint, reduce_lr_loss], verbose=1)





#     # load best model weights

#     model.load_weights(filepath)



#     # predict valid data

#     valid_pred = model.predict(X_val, batch_size=32).reshape((-1,1))



#     valid_score = mean_absolute_percentage_error(y_val,  valid_pred)

#     print ('valid mape:',valid_score)



#     scores.append(valid_score)



#     print('val CV Score of Fold_%d is %f' % (i, valid_score))

#     print('---------------------------------------------------')



#     # 予測確率の平均値を求める

#     y_pred_table_avg += model.predict(X_test, batch_size=32).reshape(-1,)



#     print('===========================================================')

        

# print('val avg:', np.mean(scores))

# print(scores)



# y_pred_table = y_pred_table_avg / n_folds

# loss = history.history['loss']

# val_loss = history.history['val_loss']

# epochs = range(len(loss))

# plt.plot(epochs, loss, 'bo' ,label = 'training loss')

# plt.plot(epochs, val_loss, 'b' , label= 'validation loss')

# plt.title('Training and Validation loss')

# plt.legend()

# plt.show()
# y_pred_table = y_pred_table.reshape(-1,1)

# y_pred_table = np.expm1(y_pred_table)

# print(y_pred_table)

train = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/train.csv')

test = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/test.csv')



#train['price'] = train['price'].apply(np.log1p)
# 部屋数が多い。もしく、敷地が広いにも関わらず価格の安い物件を外れ値とし、削除する(12行)

# id 41, 22, 37, 153,      290, 389, 532, 90, 134, 235, 257, 462

train.drop(train.index[train['id'] == 41], inplace=True)

train.drop(train.index[train['id'] == 22], inplace=True)

train.drop(train.index[train['id'] == 37], inplace=True) 

train.drop(train.index[train['id'] == 153], inplace=True)

train.drop(train.index[train['id'] == 290], inplace=True)

train.drop(train.index[train['id'] == 389], inplace=True)

train.drop(train.index[train['id'] == 532], inplace=True)

train.drop(train.index[train['id'] == 90], inplace=True)

train.drop(train.index[train['id'] == 134], inplace=True) 

train.drop(train.index[train['id'] == 235], inplace=True)

train.drop(train.index[train['id'] == 257], inplace=True) 

train.drop(train.index[train['id'] == 462], inplace=True)
train.shape
inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

# 画像読み込み

image = cv2.imread(inputPath+'1_bathroom.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

display(image.shape)

display(image[0][0])

# 画像を表示

plt.figure(figsize=(8,4))

plt.imshow(image)

# im_v = cv2.vconcat([image, image])

# display(im_v.shape)

# display(im_v[0][0])

# # 画像を表示

# plt.figure(figsize=(8,4))

# plt.imshow(im_v)
def load_images(df,inputPath,size,roomType):

    images = []

    for i in df['id']:

        basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType)])

        housePaths = sorted(list(glob.glob(basePath)))

        for housePath in housePaths:

            image = cv2.imread(housePath)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

        images.append(image)

    return np.array(images) / 255.0

trainInputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

testInputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images/'



size = 64



roomTypes = ['frontal', 'bathroom', 'bedroom', 'kitchen']



for roomType in roomTypes:

    exec("train_images_" + roomType + " = load_images(train,trainInputPath,size,roomType)")

    exec("test_images_" + roomType + " = load_images(test,testInputPath,size,roomType)")

roomTypes = ['frontal', 'bathroom', 'bedroom', 'kitchen']



for roomType in roomTypes:

    print(roomType)

    exec("print(train_images_" + roomType + ".shape)")

    exec("print(test_images_" + roomType + ".shape)")
# X_train = np.zeros((size*2, size*2, 3))



# X_train[0:size, 0:size] = train_images_frontal[0]

# X_train[0:size, size:size*2] = train_images_bathroom[1]

# X_train[size:size*2, size:size*2] = train_images_bedroom[2]

# X_train[size:size*2, 0:size] = train_images_kitchen[3]



# X_test = np.zeros((size*2, size*2, 3))



# X_test[0:size, 0:size] = test_images_frontal[0]

# X_test[0:size, size:size*2] = test_images_bathroom[1]

# X_test[size:size*2, size:size*2] = test_images_bedroom[2]

# X_test[size:size*2, 0:size] = test_images_kitchen[3]



# X_train.shape, X_test.shape # 左からサンプル数、高さ、幅、RGB
# 画像結合

X_train = np.hstack([train_images_frontal, train_images_bathroom, train_images_bedroom, train_images_kitchen])

X_test = np.hstack([test_images_frontal, test_images_bathroom, test_images_bedroom, test_images_kitchen])



X_train.shape, X_test.shape # 左からサンプル数、高さ、幅、RGB
y_train = train['price'].values

y_train = y_train.reshape((-1,1))

#y_train
# # holdout法でX, Yを分割

# train_x, valid_x, train_images_x, valid_images_x = train_test_split(train, train_images, test_size=0.2)

# train_y = train_x['price'].values

# valid_y = valid_x['price'].values

# display(train_images_x.shape)

# display(valid_images_x.shape)

# display(train_y.shape)

# display(valid_y.shape)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.applications import VGG16



# from keras.optimizers import Adam

# optAdam = Adam(amsgrad=True)



# from keras.optimizers import RMSprop

# optRMSprop = RMSprop(lr=0.01)



# def vgg16_finetuning(inputShape):

#     backbone = VGG16(weights='imagenet', include_top=False, input_shape=inputShape) 

    

#     for layer in backbone.layers[:15]:

#         layer.trainable = False

#     for layer in backbone.layers:

#         print("{}: {}".format(layer, layer.trainable))



#     model = Sequential(layers=backbone.layers)



#     model.add(GlobalAveragePooling2D())



#     model.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))

#     #model.add(Dropout(0.1))

#     model.add(Dense(units=64, activation='relu',kernel_initializer='he_normal'))

#     model.add(Dense(units=1, activation='linear'))



#     model.compile(loss='mape', optimizer="adam", metrics=['mape']) 

#     model.summary()

#     return model
# filepath = "data_aug_best_model.hdf5" 



# y_pred_image_avg = np.zeros(len(X_test))



# n_folds = 5



# scores = []



# for random_state in range(n_folds):

#     print("Training on Fold: ",random_state+1)

#     X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.1, 

#                                                         random_state = random_state, shuffle=True)

#     seed_everything(random_state)



#     # 訓練実行

#     inputShape = (size, size * 4, 3)

#     model = vgg16_finetuning(inputShape)



#     es = EarlyStopping(patience=5, mode='min', verbose=1) 



#     checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

#     reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min', factor=0.01)



#     # 訓練実行

#     datagen = ImageDataGenerator(

#                                  horizontal_flip=True,

# #                                 vertical_flip=True,

# #                                 rotation_range=90,

#                                  width_shift_range=0.2,

# #                                 height_shift_range=0.2,

#                                  zoom_range=0.3,

# #                                 shear_range=0.1,

#                                  )



    

#     datagen.fit(X_train_, augment=True)

#     train_datagen = datagen.flow(X_train_, y_train_, batch_size=32, shuffle=True)



#     batch_size = 32



#     history = model.fit(train_datagen, validation_data=(X_val, y_val),

#         steps_per_epoch=len(X_train_) / batch_size, epochs=100,                

#         callbacks=[es, checkpoint, reduce_lr_loss])



#     # load best model weights

#     model.load_weights(filepath)



#     # 評価

#     valid_pred = model.predict(X_val, batch_size=batch_size)

#     mape_score = mean_absolute_percentage_error(y_val, valid_pred)



#     scores.append(mape_score)

#     print(mape_score)



# #   乱数を変えた予測結果の加算

#     exec("y_pred_" + str(random_state) + " = model.predict(X_test, batch_size=batch_size)")

#     print('===========================================================')

        

# print(np.mean(scores))

# print(scores)
# loss = history.history['loss']

# val_loss = history.history['val_loss']

# epochs = range(len(loss))

# plt.plot(epochs, loss, 'bo' ,label = 'training loss')

# plt.plot(epochs, val_loss, 'b' , label= 'validation loss')

# plt.title('Training and Validation loss')

# plt.legend()

# plt.show()
def create_cnn(inputShape):

    model = Sequential()



    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',

                     activation='relu', kernel_initializer='he_normal', input_shape=inputShape))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', 

                     activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', 

                     activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', 

                     activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    

    model.add(GlobalAveragePooling2D())

    

    model.add(Flatten())

    

    model.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))  

    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=1, activation='linear'))

    

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    return model
scores = []



filepath = "data_aug_best_model.hdf5" 

y_pred_image_avg = np.zeros(len(X_test))

n_folds = 5



for random_state in range(n_folds):

    print("Training on Fold: ",random_state+1)

    X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, 

                                                        random_state = random_state, shuffle=True)

    seed_everything(random_state)



    # callback parameter

    filepath = "cnn_best_model.hdf5" 

    es = EarlyStopping(patience=5, mode='min', verbose=1) 

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



    # 訓練実行

    inputShape = (size, size * 4, 3)

    model = create_cnn(inputShape)

    history = model.fit(X_train_, y_train_, validation_data=(X_val, y_val),epochs=30, batch_size=32,

        callbacks=[es, checkpoint, reduce_lr_loss])

    

    valid_pred = model.predict(X_val, batch_size=32)

    mape_score = mean_absolute_percentage_error(y_val, valid_pred)

    scores.append(mape_score)

    print(mape_score)



    # 乱数を変えた予測結果の加算

    exec("y_pred_" + str(random_state) + " = model.predict(X_test, batch_size=32)")

    print('===========================================================')

        

print(np.mean(scores))

print(scores)
#seed averaging

y_pred_image_avg = (

            y_pred_0 + 

            y_pred_1 + 

            y_pred_2 + 

            y_pred_3 + 

            y_pred_4 

#             y_pred_5 + 

#             y_pred_6 + 

#             y_pred_7 + 

#             y_pred_8 + 

#             y_pred_9 + 

#             y_pred_10 + 

#             y_pred_11 + 

#             y_pred_12 + 

#             y_pred_13 + 

#             y_pred_14 + 

#             y_pred_15 + 

#             y_pred_16 + 

#             y_pred_17 + 

#             y_pred_18 + 

#             y_pred_19 +

#             y_pred_20 + 

#             y_pred_21 + 

#             y_pred_22 + 

#             y_pred_23 + 

#             y_pred_24 +

#             y_pred_25 + 

#             y_pred_26 + 

#             y_pred_27 + 

#             y_pred_28 + 

#             y_pred_29

                    ) / n_folds
y_pred_image_avg
#pred_test = 0.5 * y_pred_image_avg + 0.5 * y_pred_table_avg

pred_test = y_pred_image_avg

pred_test
np.average(pred_test)
!rm -rf *hdf5
ls -alh
submission = pd.read_csv('/kaggle/input/4th-datarobot-ai-academy-deep-learning/sample_submission.csv', index_col=0)



submission.price = pred_test

submission.to_csv('submission.csv')
submission