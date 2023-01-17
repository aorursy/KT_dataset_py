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
import zipfile

import h5py

import cv2

import glob, os

from matplotlib import pyplot as plt

import h5py

import numpy as np

from tqdm import tqdm

lbl=[]

img=np.zeros((3064,224,224))

for i in range(1,3065):

    try:

        path='/kaggle/input/gurfsdhuwi/brainTumorDataPublic_1766/'

        with h5py.File(path+str(i)+'.mat') as f:

          images = f['cjdata']

          resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )

          x=np.asarray(resized)

          x=(x-np.min(x))/(np.max(x)-np.min(x))

          x=x.reshape((1,224,224))

          img[i-1]=x

          lbl.append(int(images['label'][0]))

    except:

        try:

          path='/kaggle/input/gurfsdhuwi/brainTumorDataPublic_22993064/'

          with h5py.File(path+str(i)+'.mat') as f:

              images = f['cjdata']

              resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )

              x=np.asarray(resized)

              x=(x-np.min(x))/(np.max(x)-np.min(x))

              x=x.reshape((1,224,224))

              img[i-1]=x

              lbl.append(int(images['label'][0]))

        except:

            try:

              path='/kaggle/input/gurfsdhuwi/brainTumorDataPublic_15332298/'

              with h5py.File(path+str(i)+'.mat') as f:

                  images = f['cjdata']

                  resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )

                  x=np.asarray(resized)

                  x=(x-np.min(x))/(np.max(x)-np.min(x))

                  x=x.reshape((1,224,224))

                  img[i-1]=x

                  lbl.append(int(images['label'][0]))

            except:

              path='/kaggle/input/gurfsdhuwi/brainTumorDataPublic_7671532/'

              with h5py.File(path+str(i)+'.mat') as f:

                  images = f['cjdata']

                  resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )

                  x=np.asarray(resized)

                  x=(x-np.min(x))/(np.max(x)-np.min(x))

                  x=x.reshape((1,224,224))

                  img[i-1]=x

                  lbl.append(int(images['label'][0]))

path='/kaggle/input/fdjbfcb/ksaafhsfh.mat'



with h5py.File(path) as f:

      data=f['cvind']

      idx=data[0]

import scipy.io

obj_arr = {}

obj_arr['images'] = img

obj_arr['label'] = lbl

obj_arr['fold']=idx

np.save('check.npy', obj_arr)
import zipfile

import h5py

from keras.optimizers import *

import cv2

from keras.utils import to_categorical

from matplotlib import pyplot as plt

import glob, os

from matplotlib import pyplot as plt

import h5py

from sklearn.metrics import accuracy_score

import numpy as np

from tqdm import tqdm

import time

import gc

from keras.applications import *

from keras.layers import *

from keras import backend as K

from keras.models import Model
path = F"check.npy" 

df=np.load(path,allow_pickle=True)

df=df.item()

df['images']=df['images'].astype(np.float32)
#shuffle samples

def unison_shuffled_copies(a, b):

    assert len(a) == len(b)

    p = np.random.permutation(len(a))

    return a[p], b[p]















#get train and test splits

def get_trn_tst(df,tst_fold):

  idx=np.asarray(df['fold'])

  y=np.asarray(df['label'])

  y-=1

  img=np.asarray(df['images'])

  img1=[]

  img1=np.asarray(img)

  del([img])

  gc.collect()

  trn_y=np.asarray(y[(idx!=tst_fold)])

  trn_img=np.asarray(img1[(idx!=tst_fold)])

  tst_y=np.asarray(y[(idx==tst_fold)])

  tst_img=img1[idx==tst_fold]

  trn_img=np.repeat(trn_img.reshape((trn_img.shape[0],224,224,1)),3,axis=3)

  tst_img=np.repeat(tst_img.reshape((tst_img.shape[0],224,224,1)),3,axis=3)

  return (trn_img.copy(),trn_y.copy()),(tst_img.copy(),tst_y.copy())
mod=densenet.DenseNet121(include_top=True, weights='imagenet')

mod.summary()
def load_model(last=True):   

  K.clear_session() 

  mod=densenet.DenseNet121(include_top=True, weights='imagenet')

  out_1=mod.layers[-2].output

  out=Dense(3,activation='softmax')(out_1)

  model=Model(inputs=mod.input,outputs=out)



  if last:

    for i in range(len(model.layers)):

        model.layers[i].trainable = False

  model.layers[-1].trainable=True

  return model
best_accuracy_last={}

final_accuracy_last={}

history_last={}

answers_last={}

predictions_last={}

predictions_last_best={}

times_last={}
import keras

import pandas as pd

from keras.applications.nasnet import NASNetMobile, preprocess_input

import imgaug as ia

from imgaug import augmenters as iaa



def rotate_image(image, angle):

  image_center = tuple(np.array(image.shape[1::-1]) / 2)

  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

  return result

def Hflip( images):

		seq = iaa.Sequential([iaa.Fliplr(1.0)])

		return seq.augment_images(images)

def Vflip( images):

		seq = iaa.Sequential([iaa.Flipud(1.0)])

		return seq.augment_images(images)

def noise(images):

    ls=[]

    for i in images:

        x = np.random.normal(loc=0, scale=0.05, size=(299,299,3))

        ls.append(i+x)

    return ls

def rotate(images):

    ls=[]

    for angle in range(-15,20,5):

        for image in images:

            ls.append(rotate_image(image,angle))

    return ls



class DataGenerator(keras.utils.Sequence):

  def __init__(self, images, labels, batch_size=64, image_dimensions = (96 ,96 ,3), shuffle=False, augment=False):

    self.labels       = labels              # array of labels

    self.images = images        # array of image paths

    self.batch_size   = batch_size          # batch size

    self.on_epoch_end()



  def __len__(self):

    return int(np.floor(self.labels.shape[0] / self.batch_size))



  def on_epoch_end(self):

    self.indexes = np.arange(self.labels.shape[0])



  def __getitem__(self, index):

		# selects indices of data for next batch

    indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

    # select data and load images

    labels = self.labels.loc[indexes]

    img = [self.images[k].astype(np.float32) for k in indexes]

    imgH=Hflip(img)

    imgV=Vflip(img)

    imgR=rotate(img)

    images=[]

    images.extend(imgH)

    images.extend(imgV)

    images.extend(imgR)

    lbl=labels.copy()

    labels=pd.DataFrame()

    labels=pd.concat([labels,lbl],0)

    labels=pd.concat([labels,lbl],0)

    labels=pd.concat([labels,lbl],0)

    labels=pd.concat([labels,lbl],0)

    labels=pd.concat([labels,lbl],0)

    labels=pd.concat([labels,lbl],0)

    labels=pd.concat([labels,lbl],0)

    labels=pd.concat([labels,lbl],0)

    labels=pd.concat([labels,lbl],0)

    #images = np.array([preprocess_input(img) for img in images])

    return np.asarray(images), labels
  def upd(dk,data):

    if dk==0:

        dk=data

    else:

        for ky in data.keys():

            dk[ky].extend(data[ky])

    return dk

  index=1

  epoch=50

  pre_acc=0

  best=0

  fold='fold_'+str(index)

  trn,tst=get_trn_tst(df,index)

  history_last[fold]=0







  plt.imshow(trn[0][0])

  plt.show()

  plt.imshow(tst[0][0])

  plt.show()







  trn_x,trn_y=unison_shuffled_copies(trn[0],trn[1])

  tst_x,tst_y=unison_shuffled_copies(tst[0],tst[1])







  model=load_model(last=False)





  

  #compiling the model

  model.compile(optimizer=Adam(3e-4,decay=1e-3), 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])

  train_data = DataGenerator(trn_x,pd.get_dummies(trn_y), batch_size=4, augment=True)



  

  #fitting the model

  #timing

  start=time.time()

  for i in range(epoch):

      hist=model.fit_generator(train_data,epochs=1,steps_per_epoch=len(trn_x)//4)

      history_last[fold]=upd(history_last[fold],hist.history)



  end=time.time()

  times_last[fold]=end-start



  #getting the prediction 

  pre=model.predict(tst_x)

  







  #select the maximum position

  pre=np.argmax(pre,1)

  predictions_last[fold]=pre



  

  

  

  #getting the accuracy

  new_acc=accuracy_score(pre,tst_y)



  





  #storing the predictions

  final_accuracy_last[fold]=new_acc

















  #storing the answers

  answers_last[fold]=tst_y

    

    

    

    

  #freeing memory

  del([trn,tst,trn_x,trn_y,tst_x,tst_y])

  gc.collect()
np.save('best_accuracy_last_fold1.npy',best_accuracy_last)

np.save('final_accuracy_last_fold1.npy',final_accuracy_last)

np.save('history_last_fold1.npy',history_last)

np.save('answers_last_fold1.npy',answers_last)

np.save('predictions_last_fold1.npy',predictions_last)

np.save('predictions_last_best_fold1.npy',predictions_last_best)

np.save('times_last_fold1.npy',times_last)
plt.plot(history_last[fold]['loss'])