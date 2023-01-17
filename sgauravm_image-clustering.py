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
from zipfile import ZipFile
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.backend import flatten
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D,UpSampling2D,Dropout,Input,MaxPooling2D,concatenate,Input,BatchNormalization,Conv2DTranspose,Flatten,Dense,Reshape
from tensorflow.keras import Sequential,Model
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
import tensorflow as tf
from sklearn.cluster import KMeans
os.listdir('../input/dogs-vs-cats')
zip_file_name = '../input/dogs-vs-cats/train.zip'
with ZipFile(zip_file_name, 'r') as zip: 
    # printing all the contents of the zip file 
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall() 
    print('Done!')
os.listdir('./')
i=17
k = img_to_array(load_img('train/'+train[i]))
fig,arr = plt.subplots(1,1)
fig.set_figheight(5)
fig.set_figwidth(5)
arr.imshow(k/255)
train[0].split('.')[0]
#Subset of training data
train_sub,_ = train_test_split(train,train_size=0.4)
len(train_sub)
#training and validation split
train_images,val_images = train_test_split(train_sub,train_size=0.8)
len(train_images)
def data_gen(dir_path_img,imgs,dims,batch_size):
    while True:
        idx = np.random.choice(np.arange(len(imgs)),batch_size)
        images =[]
        for i in idx:
            img = Image.open(dir_path_img+imgs[i])
            images.append(np.array(img.resize(dims))/255)
            
        yield np.array(images),np.array(images)
gen = data_gen('train/',train_images,(256,256),20)
#Testing generator
img,_ = next(gen)
i=5
plt.imshow(img[i])
dim = (256,256)
dim
#Encoder
inp = Input((dim[0],dim[1],3))
enc = Conv2D(64,(3,3),activation='relu',padding='same')(inp)
enc = MaxPooling2D((2,2))(enc)
enc = Conv2D(128,(3,3),activation='relu',padding='same')(enc)
enc = MaxPooling2D((2,2))(enc)
enc = Conv2D(256,(3,3),activation='relu',padding='same')(enc)
enc = MaxPooling2D((2,2))(enc)
enc = Conv2D(512,(3,3),activation='relu',padding='same')(enc)
enc = MaxPooling2D((2,2))(enc)
enc = Conv2D(512,(3,3),activation='relu',padding='same')(enc)
latent = Flatten()(enc)
latent = Dense(192,activation = 'softmax')(latent)

#Decoder
dec = Reshape((8,8,3))(latent)
dec = Conv2DTranspose(512,(3,3),strides = 2,activation = 'relu',padding='same')(dec)
dec = BatchNormalization()(dec)
dec = Conv2DTranspose(512,(3,3),strides = 2,activation = 'relu',padding='same')(dec)
dec = BatchNormalization()(dec)
dec = Conv2DTranspose(256,(3,3),strides = 2,activation = 'relu',padding='same')(dec)
dec = BatchNormalization()(dec)
dec = Conv2DTranspose(128,(3,3),strides = 2,activation = 'relu',padding='same')(dec)
dec = BatchNormalization()(dec)
dec = Conv2DTranspose(64,(3,3),strides = 2,activation = 'relu',padding='same')(dec)
decoded = Conv2D(3,(3,3),activation='sigmoid',padding='same')(dec)

#Autoencoder
ae = Model(inp,decoded)
ae.compile(optimizer = Adam(lr = 1e-5), loss = 'mse')
ae.summary()
#Adam(lr = 1e-2)


batch_size = 20
spe = len(train_images)//batch_size
gen_train = data_gen('train/',train_images,(dim[0],dim[1]),batch_size)
gen_val = data_gen('train/',val_images,(dim[0],dim[1]),batch_size)
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
ae.fit(gen_train,steps_per_epoch=spe,epochs=10 )
#, validation_data = gen_val, validation_steps=len(val_images)//batch_size
gen = data_gen('train/',val_images,(256,256),20)
test_data,_ = next(gen)
res = ae.predict(test_data)
i=7
fig=plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(res[i])
fig.add_subplot(1,2,2)
plt.imshow(test_data[i])

plt.imshow()
np.sum(res[0]!=res[3])
9*9*3
encoder = Model(inputs=ae.input,outputs=ae.get_layer('dense_4').output)
encoder.summary()
# create the new nodes for each layer in the path\
idx = 12
layer_input=Input(shape=192)
x = layer_input
for layer in ae.layers[idx:]:
    x = layer(x)

# create the model
decoder = Model(layer_input, x)
decoder.summary()
encoder.save('enc.h5')
gen = data_gen('train/',val_images,(256,256),500)
val_data_full,_ = next(gen)
X = encoder.predict(val_data_full)
len(X)
kmeans =  KMeans(n_clusters=2, max_iter=1000,n_init=20)
kmeans.fit(X)
k_lab = kmeans.labels_
grp1,grp2 = val_data_full[k_lab==1],val_data_full[k_lab==0]
fig=plt.figure(figsize=(20, 10))
columns = 6
rows = 1
for i in range(1, columns*rows +1):
    img = grp1[i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
w=20
h=10
fig=plt.figure(figsize=(20, 10))
columns = 6
rows = 1
for i in range(1, columns*rows +1):
    img = grp2[i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
