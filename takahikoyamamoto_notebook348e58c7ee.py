import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

import matplotlib.pyplot as plt

%matplotlib inline

from tensorflow.keras.applications import VGG16

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm

from sklearn.model_selection import KFold

from keras.layers import Dense, Input

from keras.models import Model

from keras.layers.merge import concatenate

from sklearn.preprocessing import StandardScaler  

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



# 乱数シード固定

# seed_everything(2020)
train = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/train.csv')

test = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/test.csv')

display(train.shape)

display(train.head())
# 特徴量エンジニアリング

train['bed*bath']=train['bedrooms']*train['bathrooms']

train['bed/bath']=train['bedrooms']/train['bathrooms']

test['bed*bath']=test['bedrooms']*test['bathrooms']

test['bed/bath']=test['bedrooms']/test['bathrooms']
train['area']=train['area'].astype(str)

test['area']=test['area'].astype(str)

train['zipcode']=train['zipcode'].astype(str)

test['zipcode']=test['zipcode'].astype(str)
# cols = ['area','zipcode']

cols = ['zipcode']

encoder = OneHotEncoder()

enc_train = encoder.fit_transform(train[cols].values)

enc_test = encoder.transform(test[cols].values)
train = pd.concat([train, enc_train], axis=1)

test = pd.concat([test, enc_test], axis=1)
# bedrooms 8以上はtestデータにはない

train = train[(train['id']!=235) & (train['id']!=257) & (train['id']!=462)].reset_index(drop=True)

display(train.shape)

display(train.head())
# 特徴量

num_cols = ['bedrooms','bathrooms', 

       'bed*bath', 'bed/bath']

target = ['price']
# 正規化

scaler = StandardScaler()

train[num_cols] = scaler.fit_transform(train[num_cols])

test[num_cols] = scaler.transform(test[num_cols])
train
test
def load_images(df,inputPath,size,roomType):

    images = []

    for i in df['id']:

        basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType)])

        housePaths = sorted(list(glob.glob(basePath)))

        for housePath in housePaths:

            # 画像前処理

            image = cv2.imread(housePath)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

        images.append(image)

    return np.array(images) / 255.0
size = 64

# load train images

roomType = 'bathroom'

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images'

train_images1 = load_images(train,inputPath,size,roomType)

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images'

test_images1 = load_images(test,inputPath,size,roomType)

display(train_images1.shape)

display(train_images1[0][0][0])
# load train images

roomType = 'bedroom'

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images'

train_images2 = load_images(train,inputPath,size,roomType)

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images'

test_images2 = load_images(test,inputPath,size,roomType)

display(train_images2.shape)

display(train_images2[0][0][0])
# load train images

roomType = 'frontal'

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images'

train_images3 = load_images(train,inputPath,size,roomType)

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images'

test_images3 = load_images(test,inputPath,size,roomType)

display(train_images3.shape)

display(train_images3[0][0][0])
# load train images

roomType = 'kitchen'

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images'

train_images4 = load_images(train,inputPath,size,roomType)

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images'

test_images4 = load_images(test,inputPath,size,roomType)

display(train_images4.shape)

display(train_images4[0][0][0])
len(train)
train_images12h =[]

for i in range(len(train)):

    train_images12h.append(cv2.hconcat([train_images1[i], train_images2[i]]))
train_images34h =[]

for i in range(len(train)):

    train_images34h.append(cv2.hconcat([train_images3[i], train_images4[i]]))
# 画像結合

train_images_list =[]

for i in range(len(train)):

    train_images_list.append(cv2.vconcat([train_images12h[i], train_images34h[i]]))
train_images=np.array(train_images_list)
display(train_images.shape)

display(train_images[0][0][0])
plt.figure(figsize=(8,4))

plt.imshow(train_images[len(train)-1])
test_images12h =[]

for i in range(106):

    test_images12h.append(cv2.hconcat([test_images1[i], test_images2[i]]))
test_images34h =[]

for i in range(106):

    test_images34h.append(cv2.hconcat([test_images3[i], test_images4[i]]))
# 画像結合

test_images_list =[]

for i in range(106):

    test_images_list.append(cv2.vconcat([test_images12h[i], test_images34h[i]]))
test_images=np.array(test_images_list)
plt.figure(figsize=(8,4))

plt.imshow(test_images[105])
train_x, valid_x, train_images_x, valid_images_x = train_test_split(train, train_images, test_size=0.2)

train_y = train_x['price'].values

valid_y = valid_x['price'].values

train_x = train_x.drop(['price','id','area','zipcode'],axis=1)

valid_x = valid_x.drop(['price','id','area','zipcode'],axis=1)

test_id = test['id']

test = test.drop(['id','area','zipcode'],axis=1)



display(train_images_x.shape)

display(valid_images_x.shape)

display(train_y.shape)

display(valid_y.shape)

train_x
valid_x
test
train_y
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def cnn_mlp(inputShape,num_shape):

    input_img=Input(shape=(inputShape))

    print(inputShape)

    input_num=Input(shape=(num_shape,))

    print(input_num)



    cnn_model = Sequential()



    # 基本のCNNモデリング

    cnn_model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same',

                     activation='relu', kernel_initializer='he_normal', input_shape=inputShape))

    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(BatchNormalization())

    cnn_model.add(Dropout(0.2))

    

    # ハイパラチューニングにより精度改善

    cnn_model.add(Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', 

                     activation='relu', kernel_initializer='he_normal'))

    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(BatchNormalization())

    cnn_model.add(Dropout(0.15))

    

    cnn_model.add(Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', 

                     activation='relu', kernel_initializer='he_normal'))

    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(BatchNormalization())

    cnn_model.add(Dropout(0.2))

    

    cnn_model.add(Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same', 

                     activation='relu', kernel_initializer='he_normal'))

    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(BatchNormalization())

    cnn_model.add(Dropout(0.2))



    cnn_model.add(Flatten())

    

    # mlp

    mlp_model = Sequential()

    mlp_model.add(Dense(units=512, input_shape = (num_shape,), 

                    kernel_initializer='he_normal',activation='relu'))    

    mlp_model.add(Dropout(0.2))

    mlp_model.add(Dense(units=256,  kernel_initializer='he_normal',activation='relu'))

    mlp_model.add(Dropout(0.2))

    mlp_model.add(Dense(units=32, kernel_initializer='he_normal', activation='relu'))     

    mlp_model.add(Dropout(0.2))

    mlp_model.add(Dense(1, activation='linear'))

    

    # functional apiをつかってテーブルと画像データをMulti Inputでモデリング

    x = concatenate([cnn_model.output, mlp_model.output])

    predictions = Dense(1, activation='linear')(x)

    

    model = Model(inputs=[cnn_model.input, mlp_model.input], outputs=predictions)

    

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    return model
def vgg16_finetuning(inputShape):

    backbone = VGG16(weights='imagenet',

                    include_top=False,

                    input_shape=inputShape)

    """

    演習:Convolution Layerの重みを全部訓練してみてください！

    """    

    

#     for layer in backbone.layers[:15]:

#         layer.trainable = False

    for layer in backbone.layers:

        print("{}: {}".format(layer, layer.trainable))

        

    model = Sequential(layers=backbone.layers)     

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=64, activation='relu',kernel_initializer='he_normal'))  

    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=1, activation='linear'))

    

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    model.summary()

    return model
# cnn_mlp



# 訓練実行

inputShape = (size*2, size*2, 3)

num_shape=train_x.shape[1]



fold=5

skf = KFold(n_splits=fold, random_state=90, shuffle=True)



y_test_cm = np.zeros((106,106)) # テストデータに対する予測格納用array

mape_score =0

pred_df2 = pd.DataFrame()

valid_df2 = pd.DataFrame() 



# CrossValidation

# 精度改善

for i, (train_ix, test_ix) in enumerate(skf.split(train_images_x, train_y)):   

    train_x_ = train_x.values[train_ix]

    train_images_x_, train_y_ = train_images_x[train_ix], train_y[train_ix]

    valid_x = train_x.values[test_ix]

    valid_images_x, valid_y = train_images_x[test_ix], train_y[test_ix]



    # callback parameter

    filepath = str(i)+"_cnn_mlp_learning_best_model.hdf5" 

    es = EarlyStopping(patience=10, mode='min', verbose=1) 

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, factor=0.1, verbose=1,  mode='min')



    model = cnn_mlp(inputShape,num_shape)

    history = model.fit([train_images_x_, train_x_], train_y_, 

                        validation_data=([valid_images_x, valid_x], valid_y),epochs=100, batch_size=16,

        callbacks=[es, checkpoint, reduce_lr_loss])



    # load best model weights

    model.load_weights(filepath)



    # 評価

    valid_pred = model.predict([valid_images_x, valid_x], batch_size=32).reshape((-1,1))

    mape_score += mean_absolute_percentage_error(valid_y, valid_pred)

    print (mean_absolute_percentage_error(valid_y, valid_pred))



    valid_df2 = pd.concat([valid_df2, pd.DataFrame(valid_pred)], axis=1)

    valid_df2 = pd.concat([valid_df2, pd.DataFrame(valid_y)], axis=1)

    y_pred = model.predict([test_images,test], batch_size=32).reshape((-1,1))

    pred_df2 = pd.concat([pred_df2, pd.DataFrame(y_pred)], axis=1)



    # Seed Average

    # 精度改善

    y_test_cm+=model.predict([test_images,test], batch_size=32).reshape((-1,1))
valid_df2
pred_df2
mape_score /= fold

y_test_cm /= fold
print(mape_score)
plt.figure(figsize=(8,4))

plt.imshow(train_images_x[0])
datagen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                             rotation_range=90,      #演習

                             width_shift_range=0.1,  #演習

                             height_shift_range=0.1, #演習

                             )



print (train_images_x.shape)

for batch in datagen.flow(train_images_x,batch_size=1):

    plt.imshow(train_images_x[0])

    plt.show() 

    plt.imshow(batch[0])

    plt.show()       

    break
# vgg16_finetuning & Data Augmentation

"""

演習:ImageDataGenerator中に新たな三つを追加してみてください！

                             rotation_range=90,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

"""    



# 訓練実行 Data Augmentation

datagen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                             rotation_range=90,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

                             )

inputShape = (size*2, size*2, 3)

batch_size = 32



fold=5

skf = KFold(n_splits=fold, random_state=80, shuffle=True)



y_test = np.zeros((106,106)) # テストデータに対する予測格納用array

mape_score =0

pred_df3 = pd.DataFrame()



# CrossValidation

# 精度改善

for i, (train_ix, test_ix) in enumerate(skf.split(train_images_x, train_y)):   

    train_images_x_, train_y_ = train_images_x[train_ix], train_y[train_ix]

    valid_images_x, valid_y = train_images_x[test_ix], train_y[test_ix]



    # callback parameter

    filepath = str(i)+"_merge_data_aug_best_model.hdf5" 

    es = EarlyStopping(patience=5, mode='min', verbose=1) 

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, factor=0.01, verbose=1,  mode='min')

    

    model = vgg16_finetuning(inputShape) # Transfer Learning

    datagen.fit(train_images_x_,augment=True)

    train_datagen = datagen.flow(train_images_x_, train_y_, batch_size=batch_size, shuffle=True)



    history = model.fit(train_datagen, validation_data=(valid_images_x, valid_y),

        steps_per_epoch=len(train_images_x_) / batch_size, epochs=100,                

        callbacks=[es, checkpoint, reduce_lr_loss])

    # load best model weights

    model.load_weights(filepath)

    # 評価

    valid_pred = model.predict(valid_images_x, batch_size=32).reshape((-1,1))

    mape_score += mean_absolute_percentage_error(valid_y, valid_pred)

    print (mean_absolute_percentage_error(valid_y, valid_pred))



    y_pred = model.predict(test_images, batch_size=32).reshape((-1,1))

    pred_df3 = pd.concat([pred_df3, pd.DataFrame(y_pred)], axis=1)

    # Seed Average

    # 精度改善

    y_test+=model.predict(test_images, batch_size=32).reshape((-1,1))
mape_score /= fold

y_test /= fold
print(mape_score)
pred_df3
# Ensenble

y_test_last = 4 * y_test_cm + y_test #best
# Ensenble

y_test_last = y_test_last / 5
submission = pd.DataFrame({

"id": test_id,

"price": y_test_last.T[0]

})

submission.to_csv('submission.csv', index=False)