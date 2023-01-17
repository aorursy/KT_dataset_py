# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage import io,feature
import matplotlib.pyplot as plt
from skimage.io import MultiImage
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import os
import shutil
from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
path_2img = "../input/prostate-cancer-grade-assessment/train_images/"
path_2mask = "../input/prostate-cancer-grade-assessment/train_label_masks/"
path_2train = "../input/prostate-cancer-grade-assessment/train.csv"
remove_index = 2227
#os.mkdir('/kaggle/working/train_imgs/')
#os.mkdir('/kaggle/working/train_imgs_canny/')
#Analysing the files
data_train = pd.read_csv(path_2train)
cols = data_train.columns
print("Shape ({0} , {1})".format(len(data_train),len(cols)))
data_train.head()
data_train = data_train.drop(data_train.index[remove_index])
data_train = data_train.reset_index()
#Helper functions
def read_img(df, idx, level=2,show = False):
    img_id = df['image_id'][idx]
    provider = df['data_provider'][idx]
    grade = df['isup_grade'][idx]
    score = df['gleason_score'][idx]
    image_path = path_2img + img_id + '.tiff'

    img_im = MultiImage(image_path)
    img_im = img_im[level]
    if(show):
        image_msk = path_2mask + img_id + '_mask.tiff'
        img_msk = MultiImage(image_msk)
        img_msk = img_msk[level]
        img_msk = img_msk[:,:,0]
        im_plot(img_im,"Level 2",level)
        im_plot(img_msk,"Level 0",level)
    return img_id,provider,grade,score,img_im
def im_plot(img,nm=''):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(nm, fontsize=20)
    return
def get_slice_index(arr, assign_val = 10000,debug = False):
    ls1 = []
    ls2 = []
    for i in range(arr.shape[0]):
        temp = np.where(arr[i,:] == False)
        if(temp[0].shape[0]==0):
            ls1.append(assign_val)
            ls2.append(assign_val)
        else:
            ls1.append(temp[0][0])
            ls2.append(temp[0][-1])
    y1 = 0
    y2 = arr.shape[0]
    for i in range(len(ls1)):
        if(ls1[i] != assign_val):
            y1 = i
            break
    for i in range(len(ls2)-1,-1,-1):
        if(ls2[i] != assign_val):
            y2 = i
            break
    x1=np.min(ls1)
    test_list = [assign_val]
    #print(ls2)
    res = [i for i in ls2 if i not in test_list] 
    
    x2=np.max(res)
    #print(ls1)
    if(debug):
        print("Boundries [x1:{0},y1:{1}],[x2:{2},y2:{3}]".format(x1,y1,x2,y2))
    return [x1,y1,x2,y2]
def crop_image(data):
    list_index = get_slice_index(data,debug = False)
    return data[list_index[1]:list_index[3], list_index[0]:list_index[2]]
def convert_2_gray(data):
    data = rgb2gray(data)
    thresh = threshold_otsu(data)
    binary = data > thresh
    #print(data)
    #print("Size in Mb : ",binary.nbytes / 1000000)
    return binary  
def delete_tree(pfad="/kaggle/working/train_imgs/"):
    shutil.rmtree(pfad)
    return 
def create_img_cropped(df,l=256,w=256):
    for i in tqdm(range(len(df))):
        id_img,_,_,_,data = read_img(df, i)
        data = convert_2_gray(data)
        crop_img = crop_image(data)
        #crop_img = resize(crop_img, (l, w))
        if(i%1000==0):
            print(crop_img.shape)
        Image.fromarray(crop_img).save("/kaggle/working/train_imgs/"+id_img+".tiff")
    return
def get_canny(df,sig=0.4):
    return feature.canny(df, sigma=sig)
def get_shannon(df):
    s_ent = []
    for i in tqdm(range(len(df))):
        id_img,_,_,_,data = read_img(df, i)
        data = convert_2_gray(data)
        crop_img = crop_image(data)
        s_ent.append(shannon_entropy(crop_img))
        
    return s_ent
def feature_engineering(df):
    df['shannon'] = get_shannon(df)
    df['shannon'] = (df['shannon']-df['shannon'].min())/(df['shannon'].max()-df['shannon'].min())
    return df

def plotLearningCurve(history,epochs):
    epoch_range = range(1,epoch+1)
    plt.plot(epoch_range,history.history['accuracy'])
    plt.plot(epoch_range,history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train','val'],loc='upper left')
    plt.show()
    epoch_range = range(1,epoch+1)
    plt.plot(epoch_range,history.history['loss'])
    plt.plot(epoch_range,history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epochs')
    plt.legend(['train','val'],loc='upper left')
    plt.show()
    return
#data_train.drop(remove_index)
create_img_cropped(data_train)
#delete_tree()

#print(data_train.shape)
#_,_,_,_,test = read_img(data_train, 2227)
#im_plot(test,nm='rgb')
#test = convert_2_gray(test)
#im_plot(test,nm='binary')
#_,crop_img = crop_image(test)
#im_plot(crop_img,nm='crop')

#os.mkdir('/kaggle/working/test/')
# remove the file

#Image.fromarray(crop_img).save("test/test.png")
#data_train.head()
#building the tensorflow model
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,MaxPool2D,Dense,Conv2D,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import gc

print(tf.__version__)
gc.collect()
X1=[]
for i in tqdm(range(len(data_train))):
        id_img,_,_,_,data = read_img(data_train, i)
        data = convert_2_gray(data)
        crop_img = crop_image(data)
        X1.append(crop_img)
X1 = np.reshape(X1.shape[0],X1.reshape[1],1)
id_img,_,_,_,data = read_img(data_train, i)
data = convert_2_gray(data)
crop_img = crop_image(data)
crop_img = resize(crop_img,(300,200))
type(crop_img)
print(crop_img)
data_train = pd.get_dummies(data=data_train, columns=['isup_grade'])
data_train = data_train.drop(['index','data_provider','gleason_score'],axis = 1)
data_train.head()
X = []
path_ = "/kaggle/working/train_imgs/"
ext = ".tiff"
for i in tqdm(range(len(data_train))):
    temp = path_ + data_train['image_id'][i]+ ext
    img = image.load_img(temp,target_size=(128,128))
    #img = rgb2gray(img)
    X.append(image.img_to_array(img)/255.0)
X = np.array(X)

Y = data_train.drop(['image_id'],axis=1)
Y = Y.to_numpy()
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.2)
X.shape
#Build CNN
model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=x_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(6,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=15,validation_data=(x_test,y_test))