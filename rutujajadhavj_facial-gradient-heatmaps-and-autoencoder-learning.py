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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm,trange
from sklearn.model_selection import train_test_split
import sklearn.metrics

from tensorflow.keras import  layers, models,  applications


df_train0 = pd.read_json('../input/deepfake/metadata0.json')
df_train1 = pd.read_json('../input/deepfake/metadata1.json')
df_train2 = pd.read_json('../input/deepfake/metadata2.json')
df_train3 = pd.read_json('../input/deepfake/metadata3.json')
df_train4 = pd.read_json('../input/deepfake/metadata4.json')
df_train5 = pd.read_json('../input/deepfake/metadata5.json')
df_train6 = pd.read_json('../input/deepfake/metadata6.json')
df_train7 = pd.read_json('../input/deepfake/metadata7.json')
df_train8 = pd.read_json('../input/deepfake/metadata8.json')
df_train9 = pd.read_json('../input/deepfake/metadata9.json')
df_train10 = pd.read_json('../input/deepfake/metadata10.json')
df_train11 = pd.read_json('../input/deepfake/metadata11.json')
df_train12 = pd.read_json('../input/deepfake/metadata12.json')
df_train13 = pd.read_json('../input/deepfake/metadata13.json')
df_train14 = pd.read_json('../input/deepfake/metadata14.json')
df_train15 = pd.read_json('../input/deepfake/metadata15.json')
df_train16 = pd.read_json('../input/deepfake/metadata16.json')
df_train17 = pd.read_json('../input/deepfake/metadata17.json')
df_train18 = pd.read_json('../input/deepfake/metadata18.json')
df_train19 = pd.read_json('../input/deepfake/metadata19.json')
df_train20 = pd.read_json('../input/deepfake/metadata20.json')
df_train21 = pd.read_json('../input/deepfake/metadata21.json')
df_train22 = pd.read_json('../input/deepfake/metadata22.json')
df_train23 = pd.read_json('../input/deepfake/metadata23.json')
df_train24 = pd.read_json('../input/deepfake/metadata24.json')
df_train25 = pd.read_json('../input/deepfake/metadata25.json')
df_train26 = pd.read_json('../input/deepfake/metadata26.json')
df_train27 = pd.read_json('../input/deepfake/metadata27.json')
df_train28 = pd.read_json('../input/deepfake/metadata28.json')
df_train29 = pd.read_json('../input/deepfake/metadata29.json')
df_train30 = pd.read_json('../input/deepfake/metadata30.json')
df_train31 = pd.read_json('../input/deepfake/metadata31.json')
df_train32 = pd.read_json('../input/deepfake/metadata32.json')
df_train33 = pd.read_json('../input/deepfake/metadata33.json')
df_train34 = pd.read_json('../input/deepfake/metadata34.json')
df_train35 = pd.read_json('../input/deepfake/metadata35.json')
df_train36 = pd.read_json('../input/deepfake/metadata36.json')
df_train37 = pd.read_json('../input/deepfake/metadata37.json')
df_train38 = pd.read_json('../input/deepfake/metadata38.json')
df_train39 = pd.read_json('../input/deepfake/metadata39.json')
df_train40 = pd.read_json('../input/deepfake/metadata40.json')
df_train41 = pd.read_json('../input/deepfake/metadata41.json')
df_train42 = pd.read_json('../input/deepfake/metadata42.json')
df_train43 = pd.read_json('../input/deepfake/metadata43.json')
df_train44 = pd.read_json('../input/deepfake/metadata44.json')
df_train45 = pd.read_json('../input/deepfake/metadata45.json')
df_train46 = pd.read_json('../input/deepfake/metadata46.json')
df_val1 = pd.read_json('../input/deepfake/metadata47.json')
df_val2 = pd.read_json('../input/deepfake/metadata48.json')
df_val3 = pd.read_json('../input/deepfake/metadata49.json')
df_trains = [df_train0 ,df_train1, df_train2, df_train3, df_train4,
             df_train5, df_train6, df_train7, df_train8, df_train9,df_train10,
            df_train11, df_train12, df_train13, df_train14, df_train15,df_train16, 
            df_train17, df_train18, df_train19, df_train20, df_train21, df_train22, 
            df_train23, df_train24, df_train25, df_train26, df_train27, df_train28, 
            df_train29, df_train30, df_train31, df_train32, df_train33, df_train34,
            df_train34, df_train35, df_train36, df_train37, df_train38, df_train39,
            df_train40, df_train41, df_train42, df_train43, df_train44, df_train45,
            df_train46]
df_vals=[df_val1, df_val2, df_val3]
nums = list(range(len(df_trains)+1))
LABELS = ['REAL','FAKE']
val_nums=[47, 48, 49]
df_trains
def get_path(num,x):
    num=str(num)
    if len(num)==2:
        path='../input/deepfake/DeepFake'+num+'/DeepFake'+num+'/' + x.replace('.mp4', '') + '.jpg'
    else:
        path='../input/deepfake/DeepFake0'+num+'/DeepFake0'+num+'/' + x.replace('.mp4', '') + '.jpg'
    if not os.path.exists(path):
       raise Exception
    return path
paths=[]
y=[]
for df_train,num in tqdm(zip(df_trains,nums),total=len(df_trains)):
    images = list(df_train.columns.values)
    for x in images:
        try:
            paths.append(get_path(num,x))
            y.append(LABELS.index(df_train[x]['label']))
        except Exception as err:
            #print(err)
            pass

val_paths=[]
val_y=[]
for df_val,num in tqdm(zip(df_vals,val_nums),total=len(df_vals)):
    images = list(df_val.columns.values)
    for x in images:
        try:
            val_paths.append(get_path(num,x))
            val_y.append(LABELS.index(df_val[x]['label']))
        except Exception as err:
            #print(err)
            pass

print('There are '+str(y.count(1))+' fake train samples')
print('There are '+str(y.count(0))+' real train samples')
print('There are '+str(val_y.count(1))+' fake val samples')
print('There are '+str(val_y.count(0))+' real val samples')
import random
real=[]
fake=[]
for m,n in zip(paths,y):
    if n==0:
        real.append(m)
    else:
        fake.append(m)
fake=random.sample(fake,len(real))
paths,y=[],[]
for x in real:
    paths.append(x)
    y.append(0)
for x in fake:
    paths.append(x)
    y.append(1)
real=[]
fake=[]
for m,n in zip(val_paths,val_y):
    if n==0:
        real.append(m)
    else:
        fake.append(m)
fake=random.sample(fake,len(real))
val_paths,val_y=[],[]
for x in real:
    val_paths.append(x)
    val_y.append(0)
for x in fake:
    val_paths.append(x)
    val_y.append(1)
print('There are '+str(y.count(1))+' fake train samples') # 1 for fake 
print('There are '+str(y.count(0))+' real train samples') # 0 for real
print('There are '+str(val_y.count(1))+' fake val samples')
print('There are '+str(val_y.count(0))+' real val samples')
def read_img(path):
    img =  cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (28, 28))





X=[]
for img in tqdm(paths):
    
    X.append(read_img(img)) # training images 
val_X=[]
for img in tqdm(val_paths):
    val_X.append(read_img(img)) # validation images
for x_img in X[:10]:
    print (x_img.shape)
    plt.subplot()
    plt.imshow(x_img)
    plt.show()

print (y[:10])
import random
def shuffle(X,y):
    new_train=[]
    for m,n in zip(X,y):
        new_train.append([m,n])
    random.shuffle(new_train)
    X,y=[],[]
    for x in new_train:
        X.append(x[0])
        y.append(x[1])
    return X,y
X,y=shuffle(X,y)
val_X,val_y=shuffle(val_X,val_y)
X_train = np.array(X)
Y_train = np.array(y)

X_val = np.array(val_X)
Y_val = np.array(val_y)
X_train,X_val = X_train / 255.0,X_val / 255.0

X_val.shape
Y_train
base_model = applications.ResNet50(input_shape=(50,50,3),include_top=False,weights='imagenet')
base_model.summary()
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2,activation='softmax'))
model.input.shape
#X_train = X_train.reshape(-1,50,50,1)
#X_val = X_val.reshape(-1,50,50,1)
from keras.utils import to_categorical
y_binary = to_categorical(Y_train,num_classes=2)
y_binary_val = to_categorical(Y_val,num_classes = 2)



y_binary.shape
import tensorflow as tf

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['Precision','Recall'])

history = model.fit(X_train,y_binary, epochs=10,validation_data=(X_val, y_binary_val))

Y_train
ind = 0
X_train_real,X_train_fake = [],[]
for y in Y_train:
    if y == 0: #real
        X_train_real.append(X_train[ind]) #12,130 real samples
    else:
        X_train_fake.append(X_train[ind]) #12,130 fake samples
    ind=ind+1
        
X_train_real_t = np.array(X_train_real).reshape((len(X_train_real), np.prod(np.array(X_train_real).shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (X_train_real_t.shape)
#print x_test.shape
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X_train_real_t, X_train_real_t,
                epochs=200,
                batch_size=256,
                shuffle=True)
ind = 0
X_val_real,X_val_fake = [],[]
for y in Y_val:
    if y == 0: #real
        X_val_real.append(X_val[ind]) #12,130 real samples
    else:
        X_val_fake.append(X_val[ind]) #12,130 fake samples
    ind=ind+1
        
X_val_real = np.array(X_val_real)
X_val_real.shape
for i in X_val_real[:5]:
    plt.imshow(i)
    plt.show()
X_val_real_t = np.array(X_val_real).reshape((len(X_val_real), np.prod(np.array(X_val_real).shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (X_val_real_t.shape)
#print x_test.shape
encoded_imgs_real = encoder.predict(X_val_real_t)
decoded_imgs = decoder.predict(encoded_imgs)
print (encoded_imgs_real.mean())
print (encoded_imgs_real.var())
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_train_real_t[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
X_train_fake_t = np.array(X_train_fake).reshape((len(X_train_fake), np.prod(np.array(X_train_fake).shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (X_train_fake_t.shape)
#print x_test.shape
X_val_fake_t = np.array(X_val_fake).reshape((len(X_val_fake), np.prod(np.array(X_val_fake).shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (X_val_fake_t.shape)
#print x_test.shape
autoencoder.fit(X_train_fake_t, X_train_fake_t,
                epochs=200,
                batch_size=256,
                shuffle=True)
encoded_imgs_fake = encoder.predict(X_train_fake_t)
decoded_imgs = decoder.predict(encoded_imgs)
print (encoded_imgs_fake.mean())
print (encoded_imgs_fake.var())
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_val_fake_t[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#z score for test samples from 2 distributions

Y_pred_encoder = []

def z_score (i,encoding):
    mean = encoding.mean()
    sd = np.sqrt(encoding.var())
    z = 1.0*(i.mean() - mean)/ sd 
    return (z)
    
for i in X_val:
    if z_score(i,encoded_imgs_real)< z_score(i,encoded_imgs_fake):
        Y_pred_encoder.append(0) # real
    else:
        Y_pred_encoder.append(1) # fake
