import numpy as np 
import pandas as pd 
import os 
import tensorflow as tf 
import keras
from tensorflow.keras import Input, Model 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, AlphaDropout, MaxPooling2D, AveragePooling2D, BatchNormalization, Concatenate, Flatten, Reshape, Add, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler 
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')
ss = pd.read_csv('../input/lish-moa/sample_submission.csv') 
def preprocess(df):
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2}) 
    del df['sig_id']
    return df

train_ftr = preprocess(train_features) 
test_ftr = preprocess(test_features)  
x_train = train_ftr.values 
x_test = test_ftr.values 
# column-wise standardization 
scalers = [] 
for i in range(3,x_train.shape[1]): 
    arr = x_train[:,i]
    arr = arr.reshape(-1,1) 
    sc = StandardScaler() 
    sc.fit(arr) 
    arr = sc.transform(arr) 
    arr = arr.reshape(arr.shape[0]) 
    x_train[:,i] = arr  
    scalers.append(sc)

for i in range(3, x_test.shape[1]): 
    sc = scalers[i-3] 
    arr = x_test[:,i] 
    arr = arr.reshape(-1,1)
    arr = sc.transform(arr) 
    arr = arr.reshape(arr.shape[0])
    x_test[:,i] = arr 

models = [] 
files = [x for x in os.listdir('../input/simple-resnet-best-models/')]
cnt = 1 
for file in files: 
    print("loading model {} ...".format(cnt))
    model = load_model(os.path.join('../input/simple-resnet-best-models/',file))
    models.append(model)
    cnt += 1
preds = [] 
for model in models: 
    pred = model.predict(x_test)
    preds.append(pred)
pred_avg = (preds[0] + preds[1] + preds[2] + preds[3] + preds[4] + preds[5] + preds[6] + preds[7] + preds[8] + preds[9])
pred_avg /= 10 
for i in range(ss.shape[0]):
    ss.iloc[i,1:] = pred_avg[i] 
ss.head(5) 
ss.to_csv('submission.csv', index = False)
