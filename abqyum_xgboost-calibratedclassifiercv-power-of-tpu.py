# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install -q efficientnet

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
import xgboost as xgb

from tqdm.auto import tqdm
from glob import glob
import time, gc
import cv2


import keras
from tensorflow.keras.layers import Input, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import pickle as pkl

 
from kaggle_datasets import KaggleDatasets
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import optimizers
import efficientnet.tfkeras as efn
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
AUTO = tf.data.experimental.AUTOTUNE

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


train= pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test= pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
sub   = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
train.head()
train.isna().sum()
train['age_approx']=train['age_approx'].fillna(value=train['age_approx'].mean()) 
#df["Age"] = df["Age"].fillna(value=df["Age"].mean())
#df=df.fillna(df.mean())
train['sex'] = train['sex'].astype("category").cat.codes +1

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype("category").cat.codes +1
train.head()
test.head()
test.isna().sum()

test['sex'] = test['sex'].astype("category").cat.codes +1
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype("category").cat.codes +1
test.head()
x_train = train[['sex', 'age_approx','anatom_site_general_challenge']]
y_train = train['target']

x_test = test[['sex', 'age_approx','anatom_site_general_challenge']]
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,StratifiedKFold 
parameters = {
    "n_estimators":[100,200,500,1000,2000],
    #"loss":["deviance"],
    "learning_rate": [0.01, 0.03, 0.05,0.1, 0.15, 0.2],
     
    'colsample_bytree': [0.1, 0.3,0.5,1.0],
     "max_depth":[3,5,10],
     "subsample":[0.1, 0.3, 0.5, 1],
    
    }
%%time
with strategy.scope():
    #skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)
    
    clf = RandomizedSearchCV(xgb.XGBClassifier(), 
                   
                       parameters, 
                       
                       #cv=skf.split(x_train,y_train) ,
                       verbose=10,
                       n_jobs=-1,
                       #random_state=1001,  
                       #shuffle=True, 
                       #scoring='roc_auc',
                       )

#now fit the model........................
#clf = xgb.XGBClassifier(n_estimators=2000, 
                        #max_depth=8, 
                        #objective='multi:softprob',
                        #seed=0,  
                        #nthread=-1, 
                        #learning_rate=0.15, 
                        #num_class = 2, 


                        #scale_pos_weight = (3254)
#%%time
with strategy.scope():
    
    
    clf.fit(x_train, y_train)
clf.best_params_
from sklearn.calibration import CalibratedClassifierCV
clf=xgb.XGBClassifier(n_estimators=2000,max_depth=3,learning_rate=0.05,colsample_bytree=0.3,subsample=1,nthread=-1)

with strategy.scope():
    
    clf.fit(x_train, y_train,verbose=True)
    clf=CalibratedClassifierCV(clf, method="sigmoid")
    clf.fit(x_train, y_train)
sub.target = clf.predict_proba(x_test)[:,1]
sub_tabular = sub.copy()
sub_tabular

sub.to_csv('submission.csv', index = False)
sub.head()