%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
torch.cuda.is_available()
torch.backends.cudnn.enabled
print(os.listdir('../input/'))
PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
arch = resnet34
sz = 14
train_df = pd.read_csv('../input/fashion-mnist_train.csv')
test_df  = pd.read_csv('../input/fashion-mnist_test.csv')

print(f'train_df shape : {train_df.shape}')
print(f'test_df shape  : {test_df.shape}')
test_df.head(10)
valid_df = train_df.sample(frac=0.2)
train_df = train_df.drop(valid_df.index)
print(f'Train_df shape : {train_df.shape}')
print(f'Valid_df shape : {valid_df.shape}')
print(f'Test_df shape  : {test_df.shape}')
def split_df(df):
    '''return a tuple (X, y) 
    
        X : the training inputs which is in (samples, height, width, channel) shape
        y : the label which is flatten
    '''
    y = df['label'].values.flatten()
    X = df.drop('label', axis=1).values
    X = X.reshape(X.shape[0], 28, 28)
    return (X,y)
X_train, y_train = split_df(train_df)
X_valid, y_valid = split_df(valid_df)
X_test, y_test   = split_df(test_df)
print(f'Train set shape : {X_train.shape, y_train.shape}')
print(f'Valid set shape : {X_valid.shape, y_valid.shape}')
print(f'Test  set shape  : {X_test.shape, y_test.shape}')
# normalizing data 
X_train = X_train.astype('float64') / 255
X_valid = X_valid.astype('float64') / 255
X_test = X_test.astype('float64') / 255
# adding missing color channels 
X_train = np.stack((X_train,) * 3, axis=-1)
X_valid = np.stack((X_valid,) * 3, axis=-1)
X_test  = np.stack((X_test,) * 3, axis=-1)
print(f'Train set shape : {X_train.shape, y_train.shape}')
print(f'Valid set shape : {X_valid.shape, y_valid.shape}')
print(f'Test  set shape  : {X_test.shape, y_test.shape}')
labels = ['T-shirt/top',"Trouser","Pullover","Dress","Coat","Sandal","Shirt",'Sneaker',"Bag","Ankle boot"]
index = 3
plt.imshow(X_train[index,], cmap='gray')
plt.title(labels[y_train[index]])
data = ImageClassifierData.from_arrays(PATH, trn=(X_train,y_train), classes=[0,1,2,3,4,5,6,7,8,9],val=(X_valid, y_valid), tfms=tfms_from_model(arch, 28), test=X_test)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(7e-3, 3, cycle_len=1, cycle_mult=2)
log_preds, _ = learn.TTA(is_test=True)
prods = np.exp(log_preds)
prods = np.mean(prods, 0)
accuracy_np(prods, y_test)


