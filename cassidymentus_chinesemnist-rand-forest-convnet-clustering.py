# 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
chinese_mnist=pd.read_csv('../input/chinese-mnist/chinese_mnist.csv')
chinese_mnist
import imageio
import skimage
import skimage.io
import skimage.transform
from matplotlib import pyplot as plt
IMAGE_PATH = '..//input//chinese-mnist//data//data//'
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 1
image_files = list(os.listdir(IMAGE_PATH))
print("# of image files:{}".format(len(image_files)))
def create_file_name(x):
    file_name=f"input_{x[0]}_{x[1]}_{x[2]}.jpg"
    return file_name
import copy
data_df=copy.copy(chinese_mnist)
data_df["file"]=data_df.apply(create_file_name,axis=1)
data_df.head()
def read_image_sizes(path,file_name):
    image = skimage.io.imread(path+file_name)
    return list(image.shape)
m=np.stack(data_df['file'].apply(lambda f:read_image_sizes(IMAGE_PATH,f)))
df = pd.DataFrame(m,columns=['w','h'])
data_df=pd.concat([data_df,df],axis=1,sort=False)
    
data_df.head()
print(f"Number of suites: {data_df.suite_id.nunique()}")
print(f"Samples: {data_df.sample_id.unique()}")
num_images=len(data_df)
image_dim=64

image_array=np.zeros((num_images,image_dim,image_dim))
for im_dex in range(num_images):
    image_array[im_dex,:,:]=\
    skimage.io.imread(IMAGE_PATH+data_df['file'][im_dex])
code_array=np.array(data_df['code'])
#code_array
plt.imshow(image_array[1,:,:])
image_array[1,:,:]
import random
rand_perm=np.random.permutation(len(image_array))
image_array=image_array[rand_perm,:,:]
code_array=code_array[rand_perm]
train_im=image_array[:int(np.floor(len(image_array)/2)),:,:]
test_im=image_array[int(np.floor(len(image_array)/2)):,:,:]
train_code=code_array[:int(np.floor(len(image_array)/2))]
test_code=code_array[int(np.floor(len(image_array)/2)):]
train=np.reshape(train_im,
                 (int(np.floor(len(image_array)/2)),64**2))
test=np.reshape(test_im,
                 (int(np.ceil(len(image_array)/2)),64**2))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=300)
rfc.fit(train,train_code)
print('test result: ',rfc.score(test,test_code),'train result: ', rfc.score(train,train_code))
import random
rand_perm=np.random.permutation(len(image_array))
image_array=image_array[rand_perm,:,:]
code_array=code_array[rand_perm]
train_im=image_array[:int(np.floor(len(image_array)/2)),:,:]
test_im=image_array[int(np.floor(len(image_array)/2)):,:,:]
train_code=code_array[:int(np.floor(len(image_array)/2))]
test_code=code_array[int(np.floor(len(image_array)/2)):]

from keras.utils import to_categorical
train_images=train_im.reshape((7500,64,64,1))
train_images=train_images.astype('float32')
train_code=train_code-1
train_labels=to_categorical(train_code)

test_images=test_im.reshape((7500,64,64,1))
test_images=test_images.astype('float32')
test_code=test_code-1
test_labels=to_categorical(test_code)
np.shape(train_labels)
from keras import models, layers

model = models.Sequential()
model.add(layers.Conv2D(32,(9,9),activation='relu', 
                        input_shape=(64,64,1)))


model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(9,9),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(9,9),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(15,activation='softmax'))
#model.add(layers.Flatten())
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.summary()


model.fit(train_images,train_labels,epochs=7,batch_size=64)
test_loss,test_acc=model.evaluate(test_images,test_labels)
rand_perm=np.random.permutation(len(image_array))
image_array=image_array[rand_perm,:,:]
code_array=code_array[rand_perm]
train_im=image_array[:int(np.floor(len(image_array)/2)),:,:]
test_im=image_array[int(np.floor(len(image_array)/2)):,:,:]
train_code=code_array[:int(np.floor(len(image_array)/2))]
test_code=code_array[int(np.floor(len(image_array)/2)):]
train=np.reshape(train_im,
                 (int(np.floor(len(image_array)/2)),64**2))
test=np.reshape(test_im,
                 (int(np.ceil(len(image_array)/2)),64**2))
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
print('start kmeans')
KM_model=KMeans(init='k-means++',n_clusters=15,n_init=10)
KM_model.fit(train) 
print('end kmeans')
print('start PCA')
reduced_train=PCA(n_components=2).fit_transform(train)
print('end PCA')
plt.plot(reduced_train[:, 0], reduced_train[:, 1], 'k.', markersize=2)

h=2
x_min, x_max = reduced_train[:, 0].min() - 1, reduced_train[:, 0].max() + 1
y_min, y_max = reduced_train[:, 1].min() - 1, reduced_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

PCA_train=PCA(n_components=2).fit(train)
proj_train=PCA_train.inverse_transform(np.c_[xx.ravel(),yy.ravel()])
# Obtain labels for each point in mesh. Use last trained model.
Z = KM_model.predict(proj_train)
plt.figure(figsize=(10,10))
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
for l in range(1,16):
    plt.plot(reduced_train[train_code==l, 0], reduced_train[train_code==l, 1], 'x', markersize=5)
