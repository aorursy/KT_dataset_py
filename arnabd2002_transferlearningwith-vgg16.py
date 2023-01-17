# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from matplotlib import pyplot as plt
from keras.models import Sequential,Model,model_from_json
from keras.layers import Convolution2D,MaxPool2D,BatchNormalization,Flatten,Dense,Dropout,Input,Layer
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import image
from sklearn.metrics import confusion_matrix,classification_report,f1_score
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
from PIL import  Image as pIMG
# Any results you write to the current directory are saved as output.
os.listdir('../input/colorectal-histology-mnist/kather_texture_2016_image_tiles_5000/')
warnings.filterwarnings('ignore')
katherDir='../input/colorectal-histology-mnist/kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/'
X=list()
y=list()
for ff in os.listdir(katherDir):
    for f in os.listdir(katherDir+'/'+ff):
        file=katherDir+'/'+ff+'/'+f
        img=image.load_img(file,target_size=(100,100))
        X.append(np.array(img))
        y.append(ff.split('_')[1])
print('LOADING DONE!!')
X=np.array(X)
X=X.reshape(5000,100,100,3)
y=np.array(y)
num_classes=len(set(y))
leLabel=LabelEncoder()
y=leLabel.fit_transform(y)
y_cat=to_categorical(num_classes=num_classes,y=y)
input_shape=(X.shape[1],X.shape[2],X.shape[3])
np.shape(X),np.shape(y_cat)
#just keeping handy a list of the class names(tissue/cell types)
labelList=list()
for dd in os.listdir(katherDir):
    labelList.append(dd.split('_')[1])
labelList
weights_path='../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg16Model=VGG16(weights=weights_path,include_top=False,input_shape=(100,100,3))
#set the vgg16 layers as non-trainable
for l in vgg16Model.layers:
    l.trainable=False
#add the custom layers after the last maxpool layer of vgg16
customLayers=Flatten()(vgg16Model.output)
customLayers=Dense(1000,activation='relu')(customLayers)
customLayers=Dropout(0.20)(customLayers)
customLayers=Dense(500,activation='relu')(customLayers)
customLayers=Dropout(0.20)(customLayers)
customLayers=Dense(8,activation='softmax')(customLayers)
customModel=Model(input=vgg16Model.input,output=customLayers)
customModel.summary()
#just looking at random images
idx=np.random.randint(len(X))
plt.imshow(X[idx].reshape(100,100,3))
plt.xlabel(leLabel.inverse_transform(y[idx]))
plt.show()
#80% for train rest for test. Did not keep any for validation
X_train,X_test,y_train,y_test=train_test_split(X,y_cat,train_size=0.80,test_size=0.20,random_state=43)
np.shape(X_train),np.shape(y_train)
customModel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
epochs=80
batch_size=128
history=customModel.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,shuffle=True)
model_json = customModel.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
customModel.save_weights("model.h5")
print("Saved model to disk")
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
#prepare data for metrics
y_hat=loaded_model.predict(X_test)
y_hat=[np.argmax(xx) for xx in y_hat]
y_true=[np.argmax(xx) for xx in y_test]
print(confusion_matrix(y_pred=y_hat,y_true=y_true))
print(classification_report(y_pred=y_hat,y_true=y_true,target_names=labelList))
for i,score in enumerate(f1_score(y_pred=y_hat,y_true=y_true,average=None)):
    print(labelList[i],score)
#test random data from test set
idx=np.random.randint(len(X_test))
test=X_test[idx]
test=test.reshape(1,100,100,3)
print('PREDICTED:',leLabel.inverse_transform(np.argmax(customModel.predict(test))))
print('ACTUAL:',leLabel.inverse_transform(np.argmax(y_test[idx])))
os.listdir('../input/colorectal-histology-mnist/kather_texture_2016_larger_images_10/Kather_texture_2016_larger_images_10/')
testLargerFilesDir='../input/colorectal-histology-mnist/kather_texture_2016_larger_images_10/Kather_texture_2016_larger_images_10/'
testLargerFilesList=os.listdir(testLargerFilesDir)
from keras.preprocessing.image import image
test_img=np.array(image.load_img(testLargerFilesDir+'/'+testLargerFilesList[8]))
test_img_cpy=test_img
w,h,c=test_img.shape
subImagesList=list()
for i in range(0,w,100):
    for j in range(0,h,100):
        #print(np.shape(test_img[i:i+10,j:j+10]))
        subImagesList.append(test_img[i:i+100,j:j+100])
print(np.shape(subImagesList))
dictColor={}
dictColor['STROMA']=[0,0,0]
dictColor['DEBRIS']=[0,0,0]
dictColor['ADIPOSE']=[0,0,0]
dictColor['MUCOSA']=[0,0,0]
dictColor['EMPTY']=[0,0,0]
dictColor['TUMOR']=[255,0,0]
dictColor['LYMPHO']=[0,0,0]
dictColor['COMPLEX']=[0,0,0]
dictColor
plt.style.use('seaborn-white')
fig = plt.figure(figsize=(10,10))
for n, image in enumerate(subImagesList[:100]):
        a = fig.add_subplot(10,10, n + 1)
        category=leLabel.inverse_transform(np.argmax(loaded_model.predict(subImagesList[n].reshape(1,100,100,3))))
        cat_color=dictColor[category]
        a.text(x=0.5,y=0.5,s=category)
        codedImg=np.zeros_like(image)
        codedImg[:,:,0]=cat_color[0]
        codedImg[:,:,1]=cat_color[1]
        codedImg[:,:,2]=cat_color[2]
        plt.imshow(image,alpha=0.5)
        plt.imshow(codedImg,alpha=0.5)
        a.axis('off')
        
#fig.set_size_inches(np.array(fig.get_size_inches()) * len(subImagesList[:100]))
#plt.imshow(test_img_cpy)
plt.show()