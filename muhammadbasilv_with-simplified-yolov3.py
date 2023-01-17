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
import numpy as np

from keras.models import Sequential

import keras

import tensorflow as tf

from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,MaxPool2D,Dropout,Reshape,Add

from keras.layers import LeakyReLU

from keras.models import Model

import keras.backend as K

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

import cv2

import pandas as pd
def yolo_loss(y_true,y_pred):

  lossx=K.sum(tf.math.multiply(K.square(y_true[:,:,:,0:1]-y_pred[:,:,:,0:1]),y_true[:,:,:,4:]),axis=[1,2,3])

  lossy=K.sum(tf.math.multiply(K.square(y_true[:,:,:,1:2]-y_pred[:,:,:,1:2]),y_true[:,:,:,4:]),axis=[1,2,3])

  loss1=lossx+lossy

  lossw=K.sum(tf.math.multiply(K.square(y_true[:,:,:,2:3]-y_pred[:,:,:,2:3]),y_true[:,:,:,4:]),axis=[1,2,3])

  lossh=K.sum(tf.math.multiply(K.square(y_true[:,:,:,3:4]-y_pred[:,:,:,3:4]),y_true[:,:,:,4:]),axis=[1,2,3])

  loss2=lossw+lossh

  loss_xy_wh=loss1+loss2

  lossC=K.sum(tf.math.multiply(K.square(tf.math.subtract(y_true[:,:,:,4:],y_pred[:,:,:,4:])),y_true[:,:,:,4:]),axis=[1,2,3])

  lossC2 =K.sum(tf.math.multiply(K.square(tf.math.subtract(y_true[:,:,:,4:],y_pred[:,:,:,4:])),(1-y_true[:,:,:,4:])),axis=[1,2,3])/16  

  lossC=lossC+lossC2

    

  total_loss=loss_xy_wh+lossC

  return total_loss
yolov3=load_model("../input/trained-yolov3-model/yolov3_keras_more_epochs.hdf5",custom_objects={'yolo_loss':yolo_loss})
test_list=['../input/global-wheat-detection/test/2fd875eaa.jpg','../input/global-wheat-detection/test/348a992bb.jpg','../input/global-wheat-detection/test/51b3e36ab.jpg','../input/global-wheat-detection/test/51f1be19e.jpg','../input/global-wheat-detection/test/53f253011.jpg','../input/global-wheat-detection/test/796707dd7.jpg','../input/global-wheat-detection/test/aac893a91.jpg','../input/global-wheat-detection/test/cb8d261a3.jpg','../input/global-wheat-detection/test/cc3532ff6.jpg','../input/global-wheat-detection/test/f5a1f0358.jpg','../input/no-box-image/00b5c6764.jpg']
DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'





imagenames = os.listdir(DIR_TEST)
results=[]

for count, name in enumerate(imagenames):

    ids = name.split('.')[0]

    imagepath = '%s/%s.jpg'%(DIR_TEST,ids)

    img=cv2.imread(imagepath)

    test_img=cv2.resize(img,(256,256))

    test_img=test_img/255.0

    test_img=test_img[np.newaxis,:]

    test_box=yolov3.predict(test_img)

    box=test_box[0]

    img=cv2.resize(img,(256,256))

    pred_strings = []



    for i in range(16):

        for j in range(16):

            if(box[i][j][4]>0.8):

                x=(i*64)+(box[i][j][0]*64)

                y=(j*64)+(box[i][j][1]*64)

                w=box[i][j][2]*1024

                h=box[i][j][3]*1024

                x1=int((x-w/2)/4)

                y1=int((y-h/2)/4)

                x2=int((x+w/2)/4)

                y2=int((y+h/2)/4)

                #print(np.around(box[i][j][4],2),np.around(x1*4),np.around(y1*4),int(np.around(w)),int(np.around(h)) ,)

                #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),1)

                pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(np.clip(np.around(box[i][j][4],4),0,1), np.clip(int(np.around(x1*4)),0,1023), np.clip(int(np.around(y1*4)),0,1023), np.clip(int(np.around(w)),0,1023), np.clip(int(np.around(h)),0,1023)))

                #pred=pred+str(np.clip(np.around(box[i][j][4],4),0,1))+' '+str(np.clip(int(np.around(x1*4)),0,1023))+' '+str(np.clip(int(np.around(y1*4)),0,1023))+' '+str(np.clip(int(np.around(w)),0,1023))+' '+str(np.clip(int(np.around(h)),0,1023))+' '



    if(len(pred_strings)>0):

        result = {'image_id':ids,'PredictionString': " ".join(pred_strings)}

    else:

        result = {'image_id':ids,'PredictionString': " "}

        

    results.append(result)

    #prediction_dict[address[37:46]]=pred[:-1]

      #cv2.imwrite('/content/test.jpg',img)

    #plt.imshow(img[:,:,[2,1,0]])  

    #fig=plt.gcf()

    #fig.set_size_inches((12,12))

results
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head()
test_df.to_csv('submission.csv', index=False)