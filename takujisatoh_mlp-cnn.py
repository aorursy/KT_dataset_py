import pandas as pd
import numpy as np
import datetime
import random
import glob
import cv2
import os
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense,Input,concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model,to_categorical
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D,AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm.notebook import tqdm
%matplotlib inline

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 乱数シード固定
seed_everything(2020)
train = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/train.csv')
test = pd.read_csv('../input/4th-datarobot-ai-academy-deep-learning/test.csv')
display(train.shape)


#カテゴリ毎の平均
#summary = np.round(train.groupby(['area'])['price'].mean())
#train['area_price'] = train['area'].map(summary) 
#test['area_price'] = test['area'].map(summary) 
#summary = np.round(train.groupby(['zipcode'])['price'].mean())
#train['zipcode_price'] = train['zipcode'].map(summary) 
#test['zipcode_price'] = test['zipcode'].map(summary) 

# engeenering
train['istrain'] = 1
test['istrain'] = 0
train_test = pd.concat([train, test], axis=0)
train_test['bedxbath'] = train_test['bedrooms']* train_test['bathrooms']


#one hot encode
#train_test['bet']=train_test['bedrooms']
#train_test['bath']=train_test['bathrooms']
#train_test = pd.get_dummies(train_test, columns=['bet','bath'])

#count encode
#cols = ['area']
#for col in cols:
#    df_cnt = train_test[col].value_counts()
#    train_test[col+'_cnt'] = train_test[col].map(df_cnt)
    

## 標準化
#train_test['betMinMax']=train_test['bedrooms']
#train_test['bathMinMax']=train_test['bathrooms']
ss = StandardScaler()
train_test['bedrooms'] =ss.fit_transform(np.array(train_test['bedrooms'].values).reshape(-1,1))
train_test['bathrooms'] = ss.fit_transform(np.array(train_test['bathrooms'].values).reshape(-1,1))
#train_test['area_price'] =ss.fit_transform(np.array(train_test['area_price'].values).reshape(-1,1))
#train_test['area'] =ss.fit_transform(np.array(train_test['area'].values).reshape(-1,1))
#train_test  = train_test.drop(['zipcode'], axis=1)

train = train_test[train_test['istrain'] == 1]
test  = train_test[train_test['istrain'] == 0]
train  = train.drop(['area','zipcode','istrain'], axis=1)
test  = test.drop(['area','zipcode','price', 'istrain'], axis=1)

def load_images(df,inputPath,size):
    images = []
    for i in df['id']:
        basePath = os.path.sep.join([inputPath, "{}_*".format(i)])
        housePaths = sorted(list(glob.glob(basePath)))
        image1 = cv2.imread(housePaths[0])
        image1 = cv2.resize(image1, (size, size))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.imread(housePaths[1])
        image2 = cv2.resize(image2, (size, size))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image3 = cv2.imread(housePaths[2])
        image3 = cv2.resize(image3, (size, size))
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        image4 = cv2.imread(housePaths[3])
        image4 = cv2.resize(image4, (size, size))
        image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
        
        outputImage = np.zeros((size*2, size*2, 3))
        outputImage[0:size, 0:size] = image1
        outputImage[0:size, size:size*2] = image2
        outputImage[size:size*2, size:size*2] = image3
        outputImage[size:size*2, 0:size] = image4
        
#        plt.imshow(image)
#        plt.show() 
        images.append(outputImage)
    return np.array(images) / 255.0
def load_images_gen(df,inputPath,size):

    # ジェネレータの定義
    # データ拡張のパターンを指定する
    glasses_gen = ImageDataGenerator(rotation_range = 50)

    images = []
    for i in df['id']:
        basePath = os.path.sep.join([inputPath, "{}_*".format(i)])
        housePaths = sorted(list(glob.glob(basePath)))
        image1 = cv2.imread(housePaths[0])
        image1 = cv2.resize(image1, (size, size))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = image1[np.newaxis]
        gen1 = glasses_gen.flow(image1)
        batches1 = next(gen1)
        image1 = batches1[0]
        image2 = cv2.imread(housePaths[1])
        image2 = cv2.resize(image2, (size, size))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = image2[np.newaxis]
        gen2 = glasses_gen.flow(image2)
        batches2 = next(gen2)
        image2 = batches2[0]
        image3 = cv2.imread(housePaths[2])
        image3 = cv2.resize(image3, (size, size))
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        image3 = image3[np.newaxis]
        gen3 = glasses_gen.flow(image3)
        batches3 = next(gen3)
        image3 = batches3[0]
        image4 = cv2.imread(housePaths[3])
        image4 = cv2.resize(image4, (size, size))
        image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
        image4 = image4[np.newaxis]
        gen4 = glasses_gen.flow(image4)
        batches4 = next(gen4)
        image4 = batches4[0]

        outputImage = np.zeros((size*2, size*2, 3))
        outputImage[0:size, 0:size] = image1
        outputImage[0:size, size:size*2] = image2
        outputImage[size:size*2, size:size*2] = image3
        outputImage[size:size*2, 0:size] = image4
    #        plt.imshow(image)
    #        plt.show() 

        images.append(outputImage)
    return np.array(images) / 255.0

# load images
train_inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'
test_inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images/'
size = 32

train_images = load_images(train,train_inputPath,size)
test_images = load_images(test,test_inputPath,size)

train_images_gen = load_images_gen(train,train_inputPath,size)

train_images = np.append(train_images,train_images_gen, axis=0)
train = pd.concat([train, train])

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def create_model(numcols, inputShape):
    #### MLP ############################################

    mlp_input = Input(shape=(len(numcols),))
    mlp_hidden1 = Dense(units=512,  kernel_initializer='he_normal',activation='relu')(mlp_input)
    mlp_output   = Dropout(0.2)(mlp_hidden1)
#    mlp_hidden2 = Dense(units=256,  kernel_initializer='he_normal',activation='relu')(mlp_drop1)
#    mlp_drop2   = Dropout(0.2)(mlp_hidden2)
#    mlp_output = Dense(units=16,  kernel_initializer='he_normal',activation='relu')(mlp_hidden2)
#    mlp_output   = Dropout(0.2)(mlp_hidden3)
    #    hidden3 = Dense(units=64, kernel_initializer='he_normal', activation='relu')(drop2)
    #    output   = Dropout(0.3)(hidden3)
    #    hidden4  = Dense(units=32, kernel_initializer='he_normal', activation='relu')(drop3)
    #    output  = Dense(units=1, activation='linear')(drop3)
    #    hidden4  = Dense(units=32, kernel_initializer='he_normal', activation='relu')(drop2)
    #    output   = Dropout(0.1)(hidden4)
    #    output  = Dense(units=1, activation='linear')(drop4)    


    #### CNN ############################################

    cnn_input = Input(shape=inputShape)
    cnn_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
         activation='relu', kernel_initializer='he_normal')(cnn_input)
    cnn_pool1 = MaxPooling2D(pool_size=(2, 2))(cnn_conv1)
    cnn_btnm1 = BatchNormalization()(cnn_pool1)
    cnn_drop1 = Dropout(0.1)(cnn_btnm1)
    cnn_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', 
         activation='relu', kernel_initializer='he_normal')(cnn_drop1)
    cnn_pool2 = MaxPooling2D(pool_size=(2, 2))(cnn_conv2)
    cnn_btnm2 = BatchNormalization()(cnn_pool2)
    cnn_drop2 = Dropout(0.1)(cnn_btnm2)
    cnn_conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', 
         activation='relu', kernel_initializer='he_normal')(cnn_drop2)
    cnn_pool3 = MaxPooling2D(pool_size=(2, 2))(cnn_conv3)
    cnn_btnm3 = BatchNormalization()(cnn_pool3)
   # cnn_drop3 = Dropout(0.2)(cnn_btnm3)
    #    conv4 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', 
    #                     activation='relu', kernel_initializer='he_normal')(drop3)
    #    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #    btnm4 = BatchNormalization()(pool4)
    #    drop4 = Dropout(0.2)(btnm4)
    cnn_output = Flatten()(cnn_btnm3)
    #    hidden1 = Dense(units=256, activation='relu',kernel_initializer='he_normal')(flat)
    #   output = Dense(units=1, activation='relu',kernel_initializer='he_normal')(hidden2)
    #    output = Dense(1, activation='linear')(hidden2)
    #    hidden2 = Dense(units=128, activation='relu',kernel_initializer='he_normal')(hidden1)
    #    hidden3 = Dense(units=128, activation='relu',kernel_initializer='he_normal')(hidden1)
    #    output = Dense(1, activation='linear')(hidden3)
    
    
    #### VGG16  ############################################
    
    backbone = VGG16(weights='imagenet',include_top=False, input_shape=inputShape)
    vgg_output = Flatten()(backbone.output)    
#    vgg_hidden1 = Dense(units=512, activation='relu',kernel_initializer='he_normal')(vgg_flat) 
#    vgg_drop1 = Dropout(0.5)(vgg_hidden1)
#    vgg_hidden2 = Dense(units=32, activation='relu',kernel_initializer='he_normal')(vgg_drop1) 
#    vgg_drop2 = Dropout(0.2)(vgg_hidden2)
#    vgg_output = Dense(1, activation='linear')(vgg_drop2)

#    model = Model(inputs=backbone.input, outputs=vgg_output)


    #### concatenate #######################################

    #combined1 = concatenate([modelMLP.output,modelCNN.output,modelVGG16.output])
    combined1 = concatenate([mlp_output,cnn_output,vgg_output])
    combined2 = Dense(units=256, activation='relu',kernel_initializer='he_normal')(combined1)
    drop1 = Dropout(0.1)(combined2)
    combined3 = Dense(units=128, activation='relu',kernel_initializer='he_normal')(drop1)
    drop2 = Dropout(0.1)(combined3)
    combined4 = Dense(units=32, activation='relu',kernel_initializer='he_normal')(drop2)
    drop3 = Dropout(0.2)(combined4)
#    combined5 = Dense(units=8, activation='relu',kernel_initializer='he_normal')(drop3)
#    drop4 = Dropout(0.1)(combined5)
    combined6 = Dense(units=1, activation='linear')(drop3)

    model = Model([mlp_input]+[cnn_input]+[backbone.input], combined6)
    model.compile(loss="mape",optimizer='adam', metrics=['mape'])

    plot_model(model, to_file='funcapi.png')

    return model
def get_callbacks(filepath,es_patience,rrop_patience):
    es = EarlyStopping(patience=es_patience, mode='min', verbose=1) 
    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=rrop_patience, factor=0.1, verbose=1,  mode='min')

    return [es, checkpoint, reduce_lr_loss]
##数値カラムを取得###

cols = []
for col in test.columns:
    if test[col].dtype != 'object':
        cols.append(col)
cols.remove('id')
size2 = size * 2
inputShape = (size2, size2, 3)
batch_size = 16
epochs = 100
es_patience = 5
rrop_patience = 2
HOLD = 5
SEEDNUM = 5


train_y = train['price'].values
train_x = train.drop(['id','price'], axis=1)
seed_valid_preds = 0
seed_test_preds  = 0
seed_valid_mapes = 0

len(cv_train_x[cols])
for s in range(10, 10 + SEEDNUM):
    funcapifilepath = str(s) + "_best_model.hdf5" 
    cv_valid_preds  = np.zeros((train_x.shape[0],1))
    cv_test_preds   = np.zeros((test.shape[0],1))
    cv_valid_mapes  = 0

    folds = list(KFold(n_splits=HOLD, random_state=s*3, shuffle=True).split(train_x, train_y))

    for j, (train_idx, val_idx) in enumerate(folds):
        seed_everything(j*3)
        cv_valid_pred  = np.zeros((train_x.shape[0],1))
        cv_test_pred   = np.zeros((test.shape[0],1))
        cv_valid_mape  = 0

        print('\nFold ',j)
        cv_train_x  = train_x.iloc[train_idx]
        cv_valid_x  = train_x.iloc[val_idx]
        cv_train_images_x = train_images[train_idx]
        cv_valid_images_x = train_images[val_idx]
        cv_train_y  = train_y[train_idx]
        cv_valid_y  = train_y[val_idx]


        multimodel = create_model(cols,inputShape)

        history = multimodel.fit([cv_train_x[cols]]+[cv_train_images_x]+[cv_train_images_x], cv_train_y, batch_size=batch_size, 
                        epochs=epochs, validation_data=([cv_valid_x[cols]]+[cv_valid_images_x]+[cv_valid_images_x], cv_valid_y), 
                        callbacks=get_callbacks(funcapifilepath,es_patience,rrop_patience), verbose=1)

        # load best model weights
        multimodel.load_weights(funcapifilepath)
        
        # 評価
        cv_valid_pred[val_idx] = multimodel.predict([cv_valid_x[cols]]+[cv_valid_images_x]+[cv_valid_images_x], batch_size=batch_size).reshape((-1,1))   
        cv_valid_mape = mean_absolute_percentage_error(cv_valid_y, cv_valid_pred[val_idx])

        cv_test_pred = multimodel.predict([test[cols]]+[test_images]+[test_images], batch_size=batch_size).reshape((-1,1))

        print ("↓↓↓cv_valid_mape↓↓↓") 
        print (cv_valid_mape)
        cv_valid_preds  = cv_valid_preds + (cv_valid_pred / HOLD)
        cv_test_preds  = cv_test_preds + (cv_test_pred / HOLD)
        cv_valid_mapes  = cv_valid_mapes + (cv_valid_mape / HOLD)
    seed_test_preds  = seed_test_preds + (cv_test_preds / SEEDNUM)
    seed_valid_mapes = seed_valid_mapes + (cv_valid_mapes / SEEDNUM)
seed_test_preds
submission1 = pd.DataFrame({"id": test.id,"price": np.round(seed_test_preds,0).T[0]})
submission1.to_csv('submission1.csv', index=False)
batch_size = 16
epochs = 100
es_patience = 5
rrop_patience = 2
HOLD = 5
SEEDNUM = 5

train_x_all = np.append(train_x,test[cols], axis=0)
train_images_x_all = np.append(train_images,test_images, axis=0)
train_y_all = np.append(train_y,np.round(seed_test_preds,0).T[0], axis=0)
seed_test_preds  = 0
seed_valid_mapes = 0
for s in range(10, 10 + SEEDNUM):
    folds = list(KFold(n_splits=HOLD, random_state=s*6, shuffle=True).split(train_x_all, train_y_all))

    funcapifilepath = str(s) + "_best_model3.hdf5" 
    cv_valid_preds  = 0
    cv_test_preds   = np.zeros((test.shape[0],1))
    cv_valid_mapes  = 0
    
    
    for j, (train_idx, val_idx) in enumerate(folds):
        seed_everything(j*3)
        cv_valid_pred  = np.zeros((train_x_all.shape[0],1))
        cv_test_pred   = np.zeros((test.shape[0],1))
        cv_valid_mape  = 0

        print('\nFold ',j)
        cv_train_x  = train_x_all[train_idx]
        cv_valid_x  = train_x_all[val_idx]
        cv_train_images_x = train_images_x_all[train_idx]
        cv_valid_images_x = train_images_x_all[val_idx]
        cv_train_y  = train_y_all[train_idx]
        cv_valid_y  = train_y_all[val_idx]


        multimodel = create_model(cols,inputShape)

        history = multimodel.fit([cv_train_x]+[cv_train_images_x]+[cv_train_images_x], cv_train_y, batch_size=batch_size, 
                        epochs=epochs, validation_data=([cv_valid_x]+[cv_valid_images_x]+[cv_valid_images_x], cv_valid_y), 
                        callbacks=get_callbacks(funcapifilepath,es_patience,rrop_patience), verbose=1)

        # load best model weights
        multimodel.load_weights(funcapifilepath)
        
        # 評価
        cv_valid_pred[val_idx] = multimodel.predict([cv_valid_x]+[cv_valid_images_x]+[cv_valid_images_x], batch_size=batch_size).reshape((-1,1))   
        cv_valid_mape = mean_absolute_percentage_error(cv_valid_y, cv_valid_pred[val_idx])

        cv_test_pred = multimodel.predict([test[cols]]+[test_images]+[test_images], batch_size=batch_size).reshape((-1,1))

        print ("↓↓↓cv_valid_mape↓↓↓") 
        print (cv_valid_mape)
        cv_valid_preds  = cv_valid_preds + (cv_valid_pred / HOLD)
        cv_test_preds  = cv_test_preds + (cv_test_pred / HOLD)
        cv_valid_mapes  = cv_valid_mapes + (cv_valid_mape / HOLD)
    seed_test_preds  = seed_test_preds + (cv_test_preds / SEEDNUM)
    seed_valid_mapes = seed_valid_mapes + (cv_valid_mapes / SEEDNUM)
seed_test_preds
submission = pd.DataFrame({"id": test.id,"price": np.round(seed_test_preds,0).T[0]})
submission.to_csv('submission.csv', index=False)