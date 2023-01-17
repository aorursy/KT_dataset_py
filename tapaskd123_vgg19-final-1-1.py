# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=np.load('/kaggle/input/compression/final.npy',allow_pickle=True)

df=df.item()
import cv2

import gc

def unison_shuffled_copies(a, b):

    assert len(a) == len(b)

    p = np.random.permutation(len(a))

    return a[p], b[p]

def convert_1(arr):

  arr=np.where(arr==1,0,arr)

  arr=np.where(arr==2,1,arr)

  arr=np.where(arr==3,2,arr)

  return arr

def get_trn_tst(df,tst_fold):

  idx=np.asarray(df['fold'])

  y=convert_1(np.asarray(df['label']))

  img=np.asarray(df['images'])

  trn_y=np.asarray(y[(idx!=tst_fold)])

  trn_img=np.asarray(img[(idx!=tst_fold)])

  tst_y=np.asarray(y[(idx==tst_fold)])

  tst_img=img[idx==tst_fold]

  trn_img=np.repeat(trn_img.reshape((trn_img.shape[0],224,224,1)),3,axis=3)

  tst_img=np.repeat(tst_img.reshape((tst_img.shape[0],224,224,1)),3,axis=3)

  return (trn_img.copy(),trn_y.copy()),(tst_img.copy(),tst_y.copy())
import scipy.io

import numpy as np

from tqdm import tqdm

from keras.applications import *

from keras.models import Sequential

from keras.layers import Dense, Flatten, GlobalAveragePooling2D

import numpy as np

from keras.optimizers import *

from keras.models import Model

from keras.callbacks import LearningRateScheduler,EarlyStopping,ReduceLROnPlateau

from keras.utils import to_categorical

import gc

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

from keras.callbacks import *

import gc

import keras

from keras.layers import *

from keras import backend as K

import keras

from time import time
epoch=100

history=[]

predictions=[]

answers=[]

times=[]

final_result=[]

for index in range(1,6):

  gc.collect()

  #loading train and test folds and showing image



  trn,tst=get_trn_tst(df,index)

  plt.imshow(trn[0][0])

  plt.show()

  plt.imshow(tst[0][0])

  plt.show()



  #loading model

  mod=VGG16(include_top=True, weights='imagenet')

  out_1=mod.layers[-1].output

  out=Dense(3,activation='softmax')(out_1)

  model=Model(inputs=mod.input,outputs=out)

  #training last layer

  

  def cng(idx):

    return 0.1-(0.1-0.001)*idx/epoch

  lrs=LearningRateScheduler(cng)

  esr=EarlyStopping(patience=3,min_delta=0.1,restore_best_weights=True)

  for i in range(len(model.layers)):

    model.layers[i].trainable = False

  model.layers[-1].trainable=True

  model.compile(optimizer='adam', 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])

  start=time()

  hist=model.fit(trn[0],to_categorical(trn[1]),batch_size=32,epochs=epoch,callbacks=[lrs])

  stop=time()

  times.append(stop-start)

  history.append(hist.history)

  x_trn=model.predict(trn[0])

  x_tst=model.predict(tst[0])

  predictions.append(x_tst)

  answers.append(tst[1])

  result=accuracy_score(tst[1],np.argmax(x_tst,1))

  final_result.append(result)

  del([trn,tst])

  gc.collect()
from matplotlib import pyplot as plt

for i in range(5):

    plt.plot(history[i]['loss'])

    plt.title('loss for fold '+str(i))

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.show()
for i in range(5):

    plt.plot(history[i]['accuracy'])

    plt.title('accuarcy for fold '+str(i))

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.show()
np.mean(times)
np.mean(final_result)
from sklearn.metrics import confusion_matrix

for i in range(len(predictions)):

    pre=np.argmax(predictions[i],1)

    print(confusion_matrix(answers[i],pre))

    print()
model.summary()