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
import cv2

import gc

import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.preprocessing import image

import scipy.io

import numpy as np

from tqdm import tqdm

from keras.applications import ResNet50

from keras.models import Sequential

from keras.layers import Dense, Flatten, GlobalAveragePooling2D

import numpy as np

from keras.optimizers import *

from keras.models import Model

from keras.callbacks import LearningRateScheduler,EarlyStopping,ReduceLROnPlateau

from keras.utils import to_categorical

import time

import gc

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import seaborn as sns

from matplotlib import pyplot as plt

from keras.layers import *

from sklearn.metrics import accuracy_score

from keras.applications import VGG19

from tqdm import tqdm

from keras import backend as K

from tqdm import tqdm
df=np.load('/kaggle/input/compression/final.npy',allow_pickle=True)

df=df.item()
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

  gc.collect()

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
#store the accuracy

final_result=[]

#store history

history=[]

#store predictions

predictions=[]

#store answers

answers=[]

#store time taken

times=[]

#store best weights

best_wts={}

#store best accuracy

best_aucc={}



#loop through each fold

for index in tqdm(range(1,6)):

  best=100

  dk={'val_loss':[],'val_accuracy':[],'loss':[],'accuracy':[]}

  #set epoch

  epoch=100

  #loading train and test folds and showing image

  trn,tst=get_trn_tst(df,index)







  #show first sample of train and test fold

  plt.imshow(trn[0][0])

  plt.show()

  plt.imshow(tst[0][0])

  plt.show()







  #shuffle train and test splits

  trn_x,trn_y=unison_shuffled_copies(trn[0],trn[1])

  tst_x,tst_y=unison_shuffled_copies(tst[0],tst[1])

  





  #loading model

  K.clear_session()

  mod=VGG19(include_top=True, weights='imagenet')

  out_1=mod.layers[-2]







  out=Dense(3,activation='softmax')(out_1.output)

  model=Model(inputs=mod.input,outputs=out)

  

  

  

  #set all layers to non trainable

  for i in range(len(model.layers)):

    model.layers[i].trainable = False

  

  

  

  

  #set last layer to trainable

  model.layers[-1].trainable=True

  







  #learning rate schedular

  def cng(idx):

    return 0.01*(0.0001)**(idx/epoch)

  lrs=LearningRateScheduler(cng)









  

  #compiling the model

  model.compile(optimizer='adam', 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])

  

  

  #fitting the model 

  #timing

  for count in range(epoch):

      start=time.time()

      hist=model.fit(trn_x,to_categorical(trn_y),batch_size=32,epochs=1,callbacks=[lrs],validation_data=[tst_x,to_categorical(tst_y)])

      end=time.time()

      for i in dk.keys():

        dk[i].append(hist.history[i][0])

      cur=hist.history['val_loss'][0]

      if cur<best:

        pre=model.predict(tst_x)

        pre=np.argmax(pre,1)

        new_acc=accuracy_score(pre,tst_y)

        best_aucc[index]=new_acc

        best=cur

        del([pre,new_acc])

        gc.collect()

  times.append(end-start)





  #getting the prediction 

  pre=model.predict(tst_x)

  



    

  #store history

  history.append(dk)





  #storing the predictions

  predictions.append(pre)







  #select the maximum position

  pre=np.argmax(pre,1)



  

  

  

  #getting the accuracy

  new_acc=accuracy_score(pre,tst_y)



  

  

  #storing the new accuracy

  final_result.append(new_acc)













  #storing the answers

  answers.append(tst_y)

    

    

    

    

  #freeing memory

  del([trn,tst,trn_x,trn_y,tst_x,tst_y])

  gc.collect()
mod=VGG19(include_top=True, weights='imagenet')

mod.summary()
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

    plt.ylabel('accuracy')

    plt.show()
np.mean(list(best_aucc.values()))
print(np.mean(final_result))


from sklearn.metrics import confusion_matrix

for i in range(len(predictions)):

    pre=np.argmax(predictions[i],1)

    print(confusion_matrix(answers[i],pre))

    print()
print(np.mean(final_result))

from sklearn.metrics import confusion_matrix

for i in range(len(predictions)):

    pre=np.argmax(predictions[i],1)

    print(confusion_matrix(answers[i],pre))

    print()
np.mean(times)

mod=VGG19(include_top=True, weights='imagenet')

mod.summary()
model.summary()