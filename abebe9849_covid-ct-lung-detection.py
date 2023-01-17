from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import  ResNet50
from keras.applications.densenet import  DenseNet121
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D,Multiply,GlobalAveragePooling2D, Input,Activation, Flatten, BatchNormalization,Dropout,Concatenate,GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.optimizers import Adam,SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
num_folds=3
kf = KFold(n_splits=num_folds, shuffle=True)
import sklearn.metrics as metric
from keras.utils import np_utils
import cv2
#efficientnetのインストール
!pip install -U efficientnet
import efficientnet.keras as efn
#base_modelを選ぶ
vgg16_model=VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)
densenet121_model=DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)
efnet_0 = efn.EfficientNetB0(weights='imagenet', include_top = False)

import glob
import os
os.chdir('/kaggle/input/computed-tomography-of-lungs-datase-for-covid19')
train_posi=glob.glob('CODE19 Data/Training Data/Covid/*')
train_nega=glob.glob('CODE19 Data/Training Data/Non Covid/*')
test_posi=glob.glob('CODE19 Data/Testing Data/Covid/*')
test_nega=glob.glob('CODE19 Data/Testing Data/Non Covid/*')
#make dataFrame
import pandas as pd
train_df_posi=pd.DataFrame(columns=['filename',"covid"])
train_df_posi["filename"]=train_posi
train_df_posi["covid"]=1
train_df_nega=pd.DataFrame(columns=['filename',"covid"])
train_df_nega["filename"]=train_nega
train_df_nega["covid"]=0
train_df=pd.concat([train_df_nega,train_df_posi])

tes_df_posi=pd.DataFrame(columns=['filename',"covid"])
tes_df_posi["filename"]=test_posi
tes_df_posi["covid"]=1
tes_df_nega=pd.DataFrame(columns=['filename',"covid"])
tes_df_nega["filename"]=test_nega
tes_df_nega["covid"]=0
test_df=pd.concat([tes_df_nega,tes_df_posi])

def data_generator(data_df, batch_size=64, shape=(224,224, 3), random_state=2020):
    y = data_df["covid"].values[:]
    filenames=data_df["filename"].values[:]
    np.random.seed(random_state)
    indices = np.arange(len(filenames))
    
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            size = len(batch_idx)
            batch_files = filenames[batch_idx]
            X_batch = np.zeros((size, *shape))
            y_batch = y[batch_idx]
            
            for i, file in enumerate(batch_files):
                img = cv2.imread(file)
                img = cv2.resize(img, shape[:2])
                X_batch[i, :, :, :] = img / 255.
            yield X_batch, y_batch
def get_model_finetune(base_model,input_shape=[None,None,3], num_classes=1):
    base = base_model
    # boolでfine_tuingするか決める
    for layer in base_model.layers:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2DをGlobalMaxPooling2Dに変更...https://qiita.com/mine820/items/1e49bca6d215ce88594a 全結合層の代わりに働き軽い。
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    prediction = Dense(num_classes, activation='sigmoid')(x)

    model = Model(input=base_model.input, output=prediction)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=1e-4),
        metrics=['accuracy']
    )
    return model
import numpy as np
imgs_test=np.zeros((121,224,224,3))
i=0
x=test_df["filename"].values[:]#画像の名前が入ったndarray
print(type(x))
for i in range(121):
  img=cv2.imread(x[i])
  img = cv2.resize(img,(224,224))
  imgs_test[i, :, :, :] = img
model_test=get_model_finetune(vgg16_model)
start_weight=model_test.get_weights()
model1=get_model_finetune(vgg16_model)
model2=get_model_finetune(vgg16_model)
model3=get_model_finetune(vgg16_model)
models=[model1,model2,model3]
preds=np.zeros(test_df["covid"].values[:].shape[0])
import time
cnt=0
for train_index, eval_index in kf.split(train_df):
    cnt+=1
    print("------------------------{}fold-------------------------".format(cnt))
    tra_df, val_df = train_df.iloc[train_index], train_df.iloc[eval_index]
    model=models[cnt-1]
    start=time.time()
    train_gen=data_generator(tra_df)
    val_gen=data_generator(val_df)
    model.set_weights(start_weight)
    model.fit_generator(train_gen, steps_per_epoch=60,validation_steps=30,epochs=4,verbose=1,validation_data=val_gen)#1epochでval_genからvalidation_stepsだけ使われる
    print("かかった時間：",time.time()-start)
    """
    predict=model.predict()
    predict=np.ravel(predict
    print(predict)"""

    #predict_=np_utils.to_categorical([np.argmax(i) for i in predict]) 
    #全部0になったりする
    preds+=np.ravel(model.predict(imgs_test)/num_folds)
    del model

lb_score1=metric.roc_auc_score(test_df["covid"].values[:],preds)
print("AUC LB:",lb_score1)
from sklearn.metrics import f1_score
from tqdm import tqdm


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result
best=threshold_search(test_df["covid"].values[:],preds)
print(best)
pd.Series(np.ravel(preds)).hist()
test_df["covid"].hist()