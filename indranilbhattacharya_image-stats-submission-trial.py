

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/train.csv")

test = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/test.csv")

train.head()
sz = 128

N=16
import os

import cv2

import skimage.io

from tqdm.notebook import tqdm

import zipfile

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import gc



from sklearn.metrics import cohen_kappa_score , confusion_matrix

import lightgbm as lgb

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings("ignore")

def feature_engineering(data = train , dir_name = "train_images"):

    r_mean = []

    g_mean=[]

    b_mean = []

    r_sd = []

    g_sd = []

    b_sd = []

    

    for i in data['image_id'].values:

        img = skimage.io.MultiImage(os.path.join(f"/kaggle/input/prostate-cancer-grade-assessment/{dir_name}"+"/"+str(i)+".tiff"))[2]

    #print(img.shape)

        shape = img.shape

        pad0 = (sz-shape[0]%sz)%sz  #### horizontal padding

        pad1 = (sz-shape[1]%sz)%sz  #### vartical padding

        img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)

        img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)

        img = img.transpose(0,2,1,3,4)

        img = img.reshape(-1,sz,sz,3)

        if len(img) < N:

            img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

 

        idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]

        img = img[idxs]

        img = (img/255.0).reshape(-1,3)

        #print(img.mean(0)[0])

        r_mean.append(img.mean(0)[0])

        g_mean.append(img.mean(0)[1])

        b_mean.append(img.mean(0)[2])

    

        r_sd.append(img.std(0)[0])

        g_sd.append(img.std(0)[1])

        b_sd.append(img.std(0)[2])

        

        del img

        gc.collect()

    

    data['r_mean'] = r_mean

    data['g_mean'] = g_mean

    data['b_mean'] = b_mean



    data['r_sd'] = r_sd

    data['g_sd'] = g_sd

    data['b_sd'] = b_sd

    

    data['data_prov_ind'] = np.where(data['data_provider'] == "radboud" , 1 , 0)

    

    return data
train = feature_engineering(data = train , dir_name = "train_images")
train.head()
#https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables

from __future__ import print_function  

import sys



local_vars = list(locals().items())

for var, obj in local_vars:

    if not var.startswith('_'):

        print(var, sys.getsizeof(obj))
features = ["data_prov_ind" , 'r_mean', 'g_mean', 'b_mean', 'r_sd', 'g_sd', 'b_sd']
def quadratic_weighted_kappa(y_hat, y):

    return cohen_kappa_score(y_hat, y, weights='quadratic')
def QWK(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.rint(preds)

    score = quadratic_weighted_kappa(preds, labels)

    return ("QWK", score, True)
y = train["isup_grade"]

train = train[features]

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)
train_dataset = lgb.Dataset(X_train, y_train)

valid_dataset = lgb.Dataset(X_test, y_test)



params = {

            "objective": 'regression',

            "metric": 'rmse',

            "seed": 0,

            "learning_rate": 0.01,

            "boosting": "gbdt",

            }

        

model = lgb.train(

            params=params,

            num_boost_round=10000,

            early_stopping_rounds=100,

            train_set=train_dataset,

            valid_sets=[train_dataset, valid_dataset],

            verbose_eval=100,

            feval=QWK)
preds = model.predict(X_test, num_iteration=model.best_iteration)

preds = np.rint(preds)

preds = np.clip(preds, 0 , 5)
model.best_iteration
print("our validation score is" , quadratic_weighted_kappa(preds, y_test))
print(confusion_matrix(preds,y_test))
from __future__ import print_function  

import sys



local_vars = list(locals().items())

for var, obj in local_vars:

    if not var.startswith('_'):

        print(var, sys.getsizeof(obj))
del X_train ,X_test ,y_train ,y_test 

gc.collect()
del train_dataset, valid_dataset , train , filenames

gc.collect()
def inference (da = test , dir_path = "test_images"):

    if os.path.exists(f'../input/prostate-cancer-grade-assessment/{dir_path}'):

        print('run inference')

        

        preds = model.predict(da[features], num_iteration=500)

        preds = np.rint(preds)

        preds = np.clip(preds, 0 ,5)

        da['isup_grade'] = preds.astype(int)

        cols = ["image_id" , "isup_grade"]

        da = da[cols]

        

    return da
train = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/train.csv")
sub = inference(da = feature_engineering(data = train.head(10) , dir_name = "train_images") , dir_path = "train_images")

sub['isup_grade'] = sub['isup_grade'].astype(int)

sub.to_csv('submission.csv', index=False)

sub.head()
if os.path.exists(f'../input/prostate-cancer-grade-assessment/test_images'):

    print("still can not access the test file ?")

    sub = inference(da = feature_engineering(data = test , dir_name = "test_images") , dir_path = "test_images")

    sub['isup_grade'] = sub['isup_grade'].astype(int)

    sub.to_csv('submission.csv', index=False)

    

else:

    sub = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/sample_submission.csv")

    sub.to_csv('submission.csv', index=False)