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
import datetime
import random
import glob
import cv2
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense,concatenate,Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras import layers
from tensorflow.keras import Input
import matplotlib.pyplot as plt
%matplotlib inline

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 乱数シード固定
seed_everything(2020)
#提出ファイル読み込み
submission_sumple=pd.read_csv('/kaggle/input/4th-datarobot-ai-academy-deep-learning/sample_submission.csv')

display(submission_sumple.shape)
display(submission_sumple.head())
#学習データ読み込み
train = pd.read_csv('/kaggle/input/4th-datarobot-ai-academy-deep-learning/train.csv')
train = train.sort_values('id')
display(train.shape)
display(train.head())
display(train.dtypes)
display(train.isnull().sum())
#テストデータ読み込み
test = pd.read_csv('/kaggle/input/4th-datarobot-ai-academy-deep-learning/test.csv')
#欠測値処理

test.head()
#数値データ
num_cols=['bedrooms','bathrooms','area','zipcode']
target=['price']
#欠測値処理
train[num_cols]=train[num_cols].fillna(-99)

#ターゲットエンコーディング
summary = train.groupby(['area'],as_index=False)[target].mean()
summary = summary.rename(columns={'price': 'area_p'})
train_1 = pd.merge(train, summary, on='area', how='left')
train_1 = train_1.drop('area', axis=1)
train_1 = train_1.rename(columns={'area_p': 'area'})
test_1 = pd.merge(test, summary, on='area', how='left')
test_1 = test_1.drop('area', axis=1)
test_1 = test_1.rename(columns={'area_p': 'area'})
summary = train_1.groupby(['zipcode'],as_index=False)[target].mean()
summary = summary.rename(columns={'price': 'zipcode_p'})
train_1 = pd.merge(train_1, summary, on='zipcode', how='left')
train_1 = train_1.drop('zipcode', axis=1)
train = train_1.rename(columns={'zipcode_p': 'zipcode'})
test_1 = pd.merge(test_1, summary, on='zipcode', how='left')
test_1 = test_1.drop('zipcode', axis=1)
test = test_1.rename(columns={'zipcode_p': 'zipcode'})

# 欠損値補填
test[num_cols] = test[num_cols].fillna(-99)

# 正規化
scaler = StandardScaler()
X_all = pd.concat([train, test], axis=0)
X_all[num_cols] = scaler.fit_transform(X_all[num_cols])
train = X_all.iloc[:train.shape[0], :]
test = X_all.iloc[train.shape[0]:, :]
display(train.head())
display(test.head())
#画像読み込み
def load_images_a(df,inputPath,size):
    images = []
    for i in df['id']:
        basePath0 = os.path.sep.join([inputPath, "{}_{}*".format(i,'bathroom')])
        basePath1 = os.path.sep.join([inputPath, "{}_{}*".format(i,'bedroom')])
        basePath2 = os.path.sep.join([inputPath, "{}_{}*".format(i,'frontal')])
        basePath3 = os.path.sep.join([inputPath, "{}_{}*".format(i,'kitchen')])
        housePaths0 = sorted(list(glob.glob(basePath0)))
        housePaths1 = sorted(list(glob.glob(basePath1)))
        housePaths2 = sorted(list(glob.glob(basePath2)))
        housePaths3 = sorted(list(glob.glob(basePath3)))
        for housePath in housePaths3:
            image0 = cv2.imread(housePaths0[0])
            image1 = cv2.imread(housePaths1[0])
            image2 = cv2.imread(housePaths2[0])
            image3 = cv2.imread(housePaths3[0])
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
            image0 = cv2.resize(image0, (size, size))
            image1 = cv2.resize(image1, (size, size))
            image2 = cv2.resize(image2, (size, size))
            image3 = cv2.resize(image3, (size, size))
            image_h0 = cv2.hconcat([image0, image1])
            image_h1 = cv2.hconcat([image2, image3])
            image = cv2.vconcat([image_h0, image_h1])
        images.append(image)
    return np.array(images) / 255.0

# load train images

inputPath = '/kaggle/input/4th-datarobot-ai-academy-deep-learning/images/train_images/'
inputPath2 = '/kaggle/input/4th-datarobot-ai-academy-deep-learning/images/test_images/'
size = 32
train_images = load_images_a(train,inputPath,size)
test_images = load_images_a(test,inputPath2,size)
#画像読み込み２
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

# load train images
size = 64
roomType = 'bathroom'
train_bathroom = load_images(train,inputPath,size,roomType)
roomType = 'bedroom'
train_bedroom = load_images(train,inputPath,size,roomType)
roomType = 'frontal'
train_frontal = load_images(train,inputPath,size,roomType)
roomType = 'kitchen'
train_kitchen = load_images(train,inputPath,size,roomType)
roomType = 'bathroom'
test_bathroom = load_images(test,inputPath2,size,roomType)
roomType = 'bedroom'
test_bedroom = load_images(test,inputPath2,size,roomType)
roomType = 'frontal'
test_frontal = load_images(test,inputPath2,size,roomType)
roomType = 'kitchen'
test_kitchen = load_images(test,inputPath2,size,roomType)
from tensorflow.keras import regularizers
#モデル作成
def create_cnn(inputShape ,TableShape):
    cnn = Sequential()
    
    
    """
    演習:kernel_sizeを変更してみてください
    """    
    cnn.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='valid',
                     activation='relu', kernel_initializer='he_normal', input_shape=inputShape))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.4))
    cnn.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='valid', 
                     activation='relu', kernel_initializer='he_normal'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='valid', 
                     activation='relu', kernel_initializer='he_normal'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.4))
    """
    演習:もう一層Conv2D->MaxPooling2D->BatchNormalization->Dropoutを追加してください
    """    
    cnn.add(Flatten())
    
    cnn.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))  
    cnn.add(Dropout(0.2))
    cnn.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    
    cnn.add(Dropout(0.1))
    cnn.add(Dense(units=1, activation='linear'))
    
    """
    テーブル用
    """
    
    L1_L2 = regularizers.l1_l2(l1=0.005,l2=0.005)
    input_df=Input(shape=(len(TableShape),))
    
    nn = Sequential()
    #nn.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid',activation='relu', kernel_initializer='he_normal', input_shape=TableShape))
    #nn.add(MaxPooling2D(pool_size=(2, 2)))
    #nn.add(BatchNormalization())
    #nn.add(Dropout(0.4))
    #nn.add(Dense(units=512,input_shape=TableShape,kernel_initializer='he_normal',activation='relu'))
    #nn.add(Dropout(0.4))
    #nn.add(Dense(units=256,kernel_initializer='he_normal',activation='relu'))
    #nn.add(Dropout(0.2))
    #nn.add(Dense(units=128,kernel_initializer='he_normal',activation='relu'))
    #nn.add(Dropout(0.2))
    #nn.add(Dense(units=32,kernel_initializer='he_normal',activation='relu',kernel_regularizer=L1_L2))
    #nn.add(Dropout(0.1))
    
    nn = Dense(units=256,input_shape=(TableShape,),kernel_initializer='he_normal',activation='relu')(input_df)
    nn = Dropout(0.4)(nn)
    nn = Dense(units=128,kernel_initializer='he_normal',activation='relu')(nn)
    nn = Dropout(0.2)(nn)
    nn = Dense(units=64,kernel_initializer='he_normal',activation='relu')(nn)
    nn = Dropout(0.2)(nn)
    nn = Dense(units=32,kernel_initializer='he_normal',activation='relu',kernel_regularizer=L1_L2)(nn)
    nn = Dropout(0.1)(nn)
    nn = Model(inputs=input_df, outputs=nn)
    """
    2つのモデルを組み合わせる
    """
    merge = concatenate([cnn.output,nn.output])
    mm = Dense(units=256, activation='relu',kernel_initializer='he_normal')(merge)
    mm = Dense(units=32, activation='relu',kernel_initializer='he_normal',kernel_regularizer=L1_L2)(mm)
    mm = Dense(units=1, activation='linear')(mm)
    model = Model(inputs=[cnn.input,nn.input],outputs=mm) 
    model.compile(loss='mape', optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False), metrics=['mape'])
    
    return model

#検証データ作成
train_x, valid_x, train_images_x, valid_images_x = train_test_split(train, train_images, test_size=0.2)
train_y = train_x['price'].values
valid_y = valid_x['price'].values

valid_x=valid_x.drop(['price','id'],axis=1)
train_x=train_x.drop(['price','id'],axis=1)

display(train_x.shape)
display(valid_x.shape)
display(train_images_x.shape)
display(valid_images_x.shape)
display(train_y.shape)
display(valid_y.shape)
# callback parameter
filepath = "cnn_best_model.hdf5" 
es = EarlyStopping(patience=5, mode='min', verbose=1) 
checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')

# 訓練実行
inputShape = (size, size, 3)
tableShape = num_cols

model = create_cnn(inputShape,tableShape)
history = model.fit([train_images_x, train_x],train_y, 
                    validation_data=([valid_images_x, valid_x],valid_y),
                    epochs=100, batch_size=5,callbacks=[es, checkpoint, reduce_lr_loss])
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# load best model weights
model.load_weights(filepath)

# 評価
valid_pred = model.predict([valid_images_x,valid_x], batch_size=32).reshape((-1,1))
mape_score = mean_absolute_percentage_error(valid_y, valid_pred)
print (mape_score)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'bo' ,label = 'training loss')
plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
test_x=test.drop(['id','price'],axis=1)
model.load_weights(filepath)
submit = model.predict([test_images,test_x], batch_size=16).reshape((-1,1))
submission = pd.DataFrame({
"id": test.id,
"price": submit.T[0]
})
submission.to_csv('submission.csv', index=False)