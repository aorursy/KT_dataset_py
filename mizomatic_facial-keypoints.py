# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys
os.chdir('/kaggle/working/')
print(os.listdir("../input"))
import keras
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler
#os.chdir("../input/training")
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/training/training.csv')
#train_data.head()
images = train_data['Image']
train_data.head()

#images.head()
images_np = images.values
#print(images_np.shape)
train_data_np = train_data.values
#print(train_data_np.shape)

img = images_np[0]
#type(img)
img_ = np.array(img.split(' '), dtype= np.float32)
#len(img_)
#len(img_)
plt.imshow(img_.reshape((96,96)),cmap ='gray')
images_final = []
for x in images_np:
    img_tr = x.split(' ')
    images_final.append(np.array(img_tr,dtype = np.float32))
images_final = np.array(images_final)
print(images_final.shape)
    
images_final = images_final.reshape((-1,96,96))
print(images_final.shape)
train_data.drop(['Image'],axis = 1,inplace=True)
train_data.head()

train_data.fillna(method = 'ffill',axis = 0,inplace= True)
train_data_np_fin = train_data.values
images_final = images_final/255.0
"""cols_mean = np.nanmean(train_data_np_fin,axis = 0)
inds = np.where(np.isnan(train_data_np_fin))
train_data_np_fin[inds] = np.take(cols_mean,inds[1])
print(np.count_nonzero(np.isnan(train_data_np_fin), axis = 0))
"""
#scaler = StandardScaler()
#train_data_np_fin = scaler.fit_transform(train_data_np_fin)
print(train_data_np_fin[0])
val_split = int(0.8 * images_final.shape[0])
train_imgs = images_final[:val_split]
train_labels = train_data_np_fin[:val_split]
val_imgs = images_final[val_split:]
val_labels = train_data_np_fin[val_split:]

train_imgs = train_imgs.reshape((-1,96,96,1))
val_imgs = val_imgs.reshape((-1,96,96,1))
print(val_split)
np.count_nonzero(np.isnan(train_labels),axis = 0)
print(train_imgs.shape,train_labels.shape)
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64,3,strides = 1,padding='same',activation = 'relu',input_shape = (96,96,1)))
model.add(keras.layers.Conv2D(128,3,strides = 1,padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.advanced_activations.ReLU())
model.add(keras.layers.Conv2D(256,6,strides = 2,padding = 'valid'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.advanced_activations.ReLU())
model.add(keras.layers.Conv2D(64,6,strides = 2,padding = 'valid'))
model.add(keras.layers.advanced_activations.ReLU())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.advanced_activations.ReLU())
model.add(keras.layers.Dense(30))

#tensorboard = keras.callbacks.TensorBoard(log_dir='~/Desktop/RL/Graph',histogram_freq=0,write_graph=True,write_images=True)
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc',patience=10)]
model.compile(optimizer = keras.optimizers.Adam(lr=0.0002),loss = keras.losses.MSE, metrics = ['acc'])
history = model.fit(train_imgs,train_labels,batch_size = 128,epochs = 80,callbacks = callbacks_list,validation_data = (val_imgs,val_labels))


val_loss,train_loss = history.history['val_loss'],history.history['loss']
plt.figure(0)
plt.plot(train_loss,'-r')
plt.plot(val_loss,'-b')
test_data = pd.read_csv('../input/test/test.csv')
test_data.head()
image_id = test_data['ImageId']
test_images = test_data['Image']
test_images_np = test_images.values
test_imgs_final = []
for x in test_images_np:
    img_test = x.split(' ')
    test_imgs_final.append(np.array(img_test,dtype=np.float32))
test_imgs_final = np.array(test_imgs_final)
print(test_imgs_final.shape)
    
    
test_imgs_final = test_imgs_final.reshape((-1,96,96,1))/255.0
prediction = model.predict(test_imgs_final)
print(prediction.shape)
lookup_table = pd.read_csv('../input/IdLookupTable.csv')
lookup_table.head()
row_id = lookup_table['RowId']
features = train_data.columns.values
features
from itertools import zip_longest as izip
location = []
imageid_np = lookup_table['ImageId'].values
features_np = lookup_table['FeatureName'].values
for img,feature_id in izip(imageid_np,features_np):
    location.append(prediction[img-1,np.where(features==feature_id)])
loc = np.stack(location,axis =0)
loc.shape
loc = np.squeeze(loc)
loc[:10]
loc_pd = pd.DataFrame({'Location':loc})
loc_pd.head()
final_pd = pd.concat([row_id,loc_pd],axis = 1)
final_pd.head()
final_pd.to_csv('submission_output.csv',index = False)