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



from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense,concatenate,Reshape

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

import matplotlib.pyplot as plt

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



# 乱数シード固定

seed_everything(2020)



from tensorflow.keras.applications import VGG16

from tensorflow.keras.preprocessing.image import ImageDataGenerator





from tensorflow import keras

from tensorflow.keras import layers

#9時間に収まりきらないので３回で

cross_val_random_states = [71,26,17]

#cross_val_random_states = [71,26,17,30,9]

#cross_val_random_states = [99]
#train用CSV取得

train = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/train.csv')

display(train.shape)

display(train.head())
#test用CSV取得

test = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/test.csv')

display(test.shape)

display(test.head())
inputPath = 'train_images/'



for i in train['id']:

    basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,'frontal')])

    housePaths = sorted(list(glob.glob(basePath)))

#各種設定

size = 128 #128 best

filters_now = 32 #32

kernel_size_now =(5, 5) #(5,5) #(6, 6)はスピードも出ないし、数値も下がらない

epochs_now = 240

batch_size_now = 8 #8



#early stopping

patience_now = 2 

#-----
#⭐学習用データ取得



def concat_tile(im_list_2d):

    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])



images = []

def load_images(df,inputPath,image_size):

   

    for i in df['id']:

        ##################################

        bathroom = cv2.imread(inputPath+str(i)+'_bathroom.jpg')

        bathroom = cv2.resize(bathroom,(image_size,image_size))

        bathroom = cv2.cvtColor(bathroom, cv2.COLOR_BGR2RGB)



        bedroom = cv2.imread(inputPath+str(i)+'_bedroom.jpg')

        bedroom = cv2.resize(bedroom,(image_size,image_size))

        bedroom = cv2.cvtColor(bedroom, cv2.COLOR_BGR2RGB)



        frontal = cv2.imread(inputPath+str(i)+'_frontal.jpg')

        frontal = cv2.resize(frontal,(image_size,image_size))

        frontal = cv2.cvtColor(frontal, cv2.COLOR_BGR2RGB)

        

        kitchen = cv2.imread(inputPath+str(i)+'_kitchen.jpg')

        kitchen = cv2.resize(kitchen,(image_size,image_size))

        kitchen = cv2.cvtColor(kitchen, cv2.COLOR_BGR2RGB)

        

        #画像の結合

        image_all = concat_tile([[bathroom, bedroom],[frontal, kitchen]])

        ###################################



        #１枚１枚読み込んでnumpy配列にブチ込んでいく

        images.append(image_all)

    

    #正規化(0〜1のデータにするため)のため255で割る

    return np.array(images) / 255.0



# load train images

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'



#画像の変換後のサイズ（正方形にリサイズする、大きすぎると時間がかかる、小さすぎると精度が出ない）



#⭐⭐⭐⭐⭐

#32だとうまく行った

#size = 64

#⭐⭐⭐⭐⭐

#roomType = 'frontal'

train_images = load_images(train,inputPath,size)

#display(train_images.shape)

#display(train_images[0])

#display(train_images[0][0][0])
#⭐テスト用データ取得



images = []

def load_images(df,inputPath,image_size):

   

    for i in df['id']:

        ##################################

        bathroom = cv2.imread(inputPath+str(i)+'_bathroom.jpg')

        bathroom = cv2.resize(bathroom,(image_size,image_size))

        bathroom = cv2.cvtColor(bathroom, cv2.COLOR_BGR2RGB)



        bedroom = cv2.imread(inputPath+str(i)+'_bedroom.jpg')

        bedroom = cv2.resize(bedroom,(image_size,image_size))

        bedroom = cv2.cvtColor(bedroom, cv2.COLOR_BGR2RGB)



        frontal = cv2.imread(inputPath+str(i)+'_frontal.jpg')

        frontal = cv2.resize(frontal,(image_size,image_size))

        frontal = cv2.cvtColor(frontal, cv2.COLOR_BGR2RGB)

        

        kitchen = cv2.imread(inputPath+str(i)+'_kitchen.jpg')

        kitchen = cv2.resize(kitchen,(image_size,image_size))

        kitchen = cv2.cvtColor(kitchen, cv2.COLOR_BGR2RGB)

        

        #画像の結合

        image_all = concat_tile([[bathroom, bedroom],[frontal, kitchen]])

        ###################################



        #１枚１枚読み込んでnumpy配列にブチ込んでいく

        images.append(image_all)

    

    #正規化(0〜1のデータにするため)のため255で割る

    return np.array(images) / 255.0



# load test images

inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images/'



#roomType = 'frontal'

test_images = load_images(test,inputPath,size)

#display(test_images.shape)

#display(test_images[0])

#display(train_images[0][0][0])
#各種設定

size = 128 #128 best

filters_now = 32 #32

kernel_size_now =(5, 5) #(5,5) #(6, 6)はスピードも出ないし、数値も下がらない

epochs_now = 120

batch_size_now = 8 #8



#early stopping

patience_now = 2 

#-----
# ⭐functional apiで　cnn書き換え

def create_cnn_func(inputShape):

   

    conv = Conv2D(filters=filters_now, kernel_size=kernel_size_now, strides=(1, 1), padding='same',

                      activation='relu',kernel_initializer='he_normal', input_shape=(size*2,size*2,3))(inputShape)



    x = MaxPooling2D(pool_size=(2, 2))(conv)

    x = BatchNormalization()(x)

    x = Dropout(0.1)(x)

    #####################################

    x = Conv2D(filters=filters_now*2, kernel_size=kernel_size_now, strides=(1, 1), padding='same',

                     activation='relu', kernel_initializer='he_normal',input_shape=(size*2,size*2,3))(x)



    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.1)(x)

    #####################################

    x = Flatten()(x)

    x = Dense(units=512, activation='relu',kernel_initializer='he_normal')(x)

  



    return x
#⭐MLP Sequentialモデルを定義する



drop_num = 0.2



def mlp_func(num_cols,inputShape2):



    x = Dense(units=512, input_shape = (len(num_cols),), 

                    kernel_initializer='he_normal',activation='relu') (inputShape2) 

    

    x = Dropout(drop_num)(x)

    

    return x
#転移学習と水増し



(size*2, size*2, 3)



def vgg16_finetuning(inputShape,inputShape3):

    backbone = VGG16(weights='imagenet',

                    include_top=False,

                    input_shape=inputShape)(inputShape3)



    model = GlobalAveragePooling2D()(backbone)

    #model = GlobalAveragePooling2D()(model)

    model = Flatten()(model)

    model = Dense(units=512, activation='relu',kernel_initializer='he_normal')(model)

    



    return model

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#モデル訓練、交差検定

filepath = "cnn_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

#⭐factor=0.1

#て何の設定だろう

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience = patience_now, verbose=1,  mode='min')





#⭐⭐⭐⭐⭐⭐⭐⭐⭐

#model = mlp_func(['bedrooms', 'bathrooms', 'area', 'zipcode'])

mlp_cols = ['bedrooms', 'bathrooms', 'area', 'zipcode']

scores = []

y_pred_all = [0] * 3

history_all= [0] * 3

inputShape = (size*2, size*2, 3)



#⭐⭐⭐⭐⭐⭐⭐⭐⭐



inputShape2 = keras.Input(shape=inputShape, name="cnn")

model = create_cnn_func(inputShape2)

#---------

inputShape_mlp = keras.Input(shape=(4), name="mlp")

model_mlp = mlp_func(mlp_cols,inputShape_mlp)

#---------

inputShape3 = keras.Input(shape=inputShape, name="fine")

model_fine = vgg16_finetuning(inputShape,inputShape3)



#⭐⭐⭐⭐⭐⭐⭐⭐⭐



merged = concatenate([model,model_mlp,model_fine])



#⭐これ外だし

x = Dense(units=256,  kernel_initializer='he_normal',activation='relu')(merged)

x = Dropout(drop_num)(x)

x = Dense(units=32, kernel_initializer='he_normal', activation='relu')(x)

x = Dropout(drop_num)(x)

output = Dense(units=1, activation='linear')(x)

    

wao_encoder = keras.Model(inputs=[inputShape2,inputShape_mlp,inputShape3],outputs = output, name="dodododo_encoder")

#⭐⭐⭐⭐⭐⭐⭐⭐⭐





wao_encoder.compile(loss='mape', optimizer='adam', metrics=['mape']) 

wao_encoder.summary()



#⭐⭐⭐⭐⭐⭐⭐⭐⭐



count = 0



#⭐⭐⭐⭐⭐ 一個にして確認中

for random_val in cross_val_random_states:

    

    print(count)

    

    #############################################

    #⭐訓練、テストデータ作成



    train_x, valid_x, train_images_x, valid_images_x = train_test_split(train, train_images, test_size=0.2,random_state= random_val)

    train_y = train_x['price'].values

    valid_y = valid_x['price'].values

    

    #----------

    train_x_mlp = train_x[['bedrooms', 'bathrooms', 'area', 'zipcode']]

    valid_x_mlp = valid_x[['bedrooms', 'bathrooms', 'area', 'zipcode']]

    #----------



    #############################################

    #⭐データ水増し

    #水増しどうするか

    datagen = ImageDataGenerator(horizontal_flip=True,

                                 vertical_flip=True,

                                 #--------------------------------

                                #演習:ImageDataGenerator中に新たな三つを追加してみてください！

                                 rotation_range=30,

                                 width_shift_range=0.2,

                                 height_shift_range=0.2,

                                #----------------------------------

                                 )



    #学習データのみ水増しする、予測用テストデータは水増ししない。

    #⭐ただしkaggleのコンペの時は予測用データも水増しして何回も学習して平均を取るやり方とかもあるらしい

    datagen.fit(train_images_x,augment=True)

    #datagen.flowで水増し

    train_datagen = datagen.flow(train_images_x, train_y, batch_size=batch_size_now, shuffle=True)



    

    

    #############################################

    #⭐モデル訓練

    history = wao_encoder.fit([train_images_x, train_x_mlp,train_images_x], [train_y,train_y,train_y], validation_data=([valid_images_x,valid_x_mlp,valid_images_x], [valid_y,valid_y,valid_y]),epochs = epochs_now, batch_size = batch_size_now,

                        callbacks=[es, checkpoint, reduce_lr_loss])





    #############################################

    #⭐予測してsubmissionファイルを作成する

    #合体させていく

    valid_pred = wao_encoder.predict([test_images,test[['bedrooms', 'bathrooms', 'area', 'zipcode']],test_images],batch_size=32).reshape((-1,1))

    mape_score = mean_absolute_percentage_error(valid_y, valid_pred)

    print (mape_score)

    history_all[count] = history

    y_pred_all[count]= valid_pred

    scores.append(mape_score)

    count += 1

#kaggle_pred =  y_pred_all[0] + y_pred_all[1] + y_pred_all[2] + y_pred_all[3] + y_pred_all[4]

kaggle_pred =  y_pred_all[0] + y_pred_all[1] + y_pred_all[2] 





kaggle_pred = kaggle_pred/3





# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/sample_submission.csv', index_col=0)

#submission = pd.read_excel('sales_prediction.xlsx', index_col=0)

#submission = pd.read_csv('sample_submission.csv', index_col=0)



submission.price = kaggle_pred

submission.to_csv('submission_last_image_net.csv')
wao_encoder.summary()
plot_model(wao_encoder, to_file='mlp.png')
for history in history_all:

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.plot(epochs, loss, 'bo' ,label = 'training loss')

    plt.plot(epochs, val_loss, 'b' , label= 'validation loss')

    plt.title('Training and Validation loss')

    plt.legend()

    plt.show()