# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import ImageDataGenerator
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from PIL import Image
import os
print(os.listdir("../input"))
from matplotlib import pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Convolution2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
import keras.backend as K
# Any results you write to the current directory are saved as output.
#os.listdir('../input/vgg16/')
vgg16NoTopFile='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_dir='../input/10-monkey-species/'
train_dir=base_dir+'training/training/'
validation_dir=base_dir+'validation/'
os.listdir(base_dir)
labelDF=pd.read_csv('../input/10-monkey-species/monkey_labels.txt')
cols=[str.rstrip(str.lstrip(x)) for x in labelDF.columns]
newLabelDict={}
for old,new in zip(labelDF.columns,cols):
        newLabelDict[old]=new
labelDF.rename(columns=newLabelDict,inplace=True)
labelDF['Label']
labelCmnNameDict={}
for (lab,name) in labelDF[['Label','Common Name']].values:
    labelCmnNameDict[str.rstrip(str.lstrip(lab))]=str.rstrip(str.lstrip(name))
labelCmnNameDict
#del imgList,labelList
imgList=[]
labelList=[]
dirlist=os.listdir(train_dir)
for dr in dirlist:
    print('Label:'+dr)
    for f in os.listdir(train_dir+'/'+dr):
        #imgData=Image.open(train_dir+'/'+dr+'/'+f).convert('L')
        imgData=Image.open(train_dir+'/'+dr+'/'+f)
        imgData=imgData.resize((200,200),Image.NEAREST)
        imgData=np.array(imgData)
        #imgData=imgData.reshape((200,200,1))
        imgList.append(imgData)
        labelList.append(dr)
img_arr=imgList[0]
np.shape(img_arr)
idx=np.random.randint(len(imgList))
plt.imshow(imgList[idx])
plt.xlabel(labelCmnNameDict[labelList[idx]])
plt.show()
le=LabelEncoder()
encodedLabelList=le.fit_transform(labelList)
num_classes=len(np.unique(encodedLabelList))
X=np.array(imgList)
y=to_categorical(num_classes=num_classes,y=encodedLabelList)
np.shape(X)
vgg16=VGG16(include_top=False,input_shape=(200,200,3),weights=vgg16NoTopFile)
for l in vgg16.layers:
    l.trainable=False
vggOut=vgg16.output
myLayer=Flatten()(vggOut)
myLayer=Dense(1000,activation='relu')(myLayer)
myLayer=Dropout(0.25)(myLayer)
myLayer=Dense(500,activation='relu')(myLayer)
myLayer=Dropout(0.20)(myLayer)
myLayer=Dense(num_classes,activation='softmax')(myLayer)
transferModel=Model(inputs=vgg16.input,outputs=myLayer)
transferModel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.80,test_size=0.20,random_state=43)
np.shape(X_train)
datagen=ImageDataGenerator(rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

#datagen.fit(X_train)
losses=transferModel.fit_generator(datagen.flow(batch_size=28,x=X_train,y=y_train),verbose=1,epochs=50)
idx=np.random.randint(len(X_test))
out=transferModel.predict(X_test[idx].reshape(1,200,200,3))
print('predicted:',labelCmnNameDict[le.inverse_transform(np.argmax(out))])
print('actual:',labelCmnNameDict[le.inverse_transform(np.argmax(y_test[idx]))])
plt.imshow(X_test[idx])
plt.show()