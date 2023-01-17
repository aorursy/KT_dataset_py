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

        break



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import seaborn

import cv2

%matplotlib inline
import tensorflow as tf

import keras

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,BatchNormalization

from keras.models import Sequential,Model
df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
df.info()
df.target.value_counts()
df.diagnosis.value_counts()
df.sex.value_counts()
df.anatom_site_general_challenge.value_counts()
sample_img=10

reshape_size = 512

channel = 3
new_df = df.head(sample_img)
print(new_df.head())

print(new_df.tail())
n=0

for i in range(df.shape[0]):

    if(df['target'][i]==1):

        n+=1

        x =df.iloc[i,:]

        x = pd.DataFrame(x.values.reshape(1,-1),columns = x.index)

        new_df=pd.concat([new_df,x],axis=0)

    if(n==sample_img):

        break

    
new_df
new_df.dropna(axis=0,inplace=True)
def labelfullpath(df,train=True):

    base_path="../input/siim-isic-melanoma-classification/jpeg"

    if(train==True):

        base_path = os.path.join(base_path,"train")

    else:

        base_path = os.path.join(base_path,"test")

    fullpath = [os.path.join(base_path,img+".jpg") for img in df.image_name]

    df['fullpath'] = fullpath

    return df
new_df = labelfullpath(new_df)

new_df.head()
new_df['gender'] = [float(new_df['sex'].values[i]=='female') for i in range(sample_img*2)]
dict_anatom = {'oral/genital':0,'palms/soles':0.20,'head/neck':0.40,'upper extremity':0.60,'lower extremity':0.80,'torso':1.0}

new_df['anatom_site'] = [dict_anatom[new_df['anatom_site_general_challenge'].values[i]] for i in range(sample_img*2)]
new_df['age'] = [(new_df['age_approx'].values[i])/100.0 for i in range(sample_img*2)]
x = new_df[['gender','anatom_site','age']].values
new_df.fullpath.values[0]
plt.figure(figsize = (20,10))

num = 1

for i in range(5):

    plt.subplot(1,5,num)

    im = plt.imread(new_df.fullpath.values[i])

    plt.imshow(im)

    plt.title(im.shape)

    plt.xlabel(new_df.sex.values[i]+" "+str(new_df.age_approx.values[i])+"\n"+new_df.anatom_site_general_challenge.values[i])

    num+=1
plt.figure(figsize = (20,10))

num = 1

for i in range(5):

    plt.subplot(1,5,num)

    im = plt.imread(new_df.fullpath.values[i+5])

    plt.imshow(im)

    plt.title(im.shape)

    plt.xlabel(new_df.sex.values[i+5]+" "+str(new_df.age_approx.values[i+5])+"\n"+new_df.anatom_site_general_challenge.values[i+5])

    num+=1
def preprocessing_images(imglist,channel=1):

    image_arr =[] 

    for img in imglist:

        if(channel==1):

            i = cv2.imread(img,cv2.IMREAD_GRAYSCALE)

        else:

            i = cv2.imread(img)

        i = cv2.resize(i,(reshape_size,reshape_size))

        i = i/255.0

        image_arr.append(i)

    return np.array(image_arr)    
images = preprocessing_images(new_df.fullpath.values,channel=channel)
images.shape
shuffle_index = [i for i in range(0,2*sample_img)]

np.random.shuffle(shuffle_index)
shuffle_images = images[shuffle_index]

shuffle_labels = new_df.target.values[shuffle_index]
if(channel==1):    

    fit_images = np.expand_dims(shuffle_images,axis=3)

else:

    fit_images = shuffle_images

onehot_labels = np.array([np.eye(2)[i] for i in shuffle_labels])
print(fit_images.shape)

print(onehot_labels.shape)
plt.imshow(shuffle_images[1],cmap = 'hot')
def simple_model():

    keras.backend.clear_session()

    

    ip1 = keras.layers.Input(shape = (reshape_size,reshape_size,channel))

    ip2 = keras.layers.Input(shape = (3,))

    vgg = keras.applications.VGG19(input_shape=(reshape_size,reshape_size,channel),include_top=False,weights = 'imagenet')(ip1)

    vgg.trainable = False

    flat = Flatten()(vgg)

    Dense1 =  Dense(525,activation='relu')(flat)

    Dense2 = Dense(525,activation='relu')(Dense1)

    Dense3 = Dense(50,activation='relu')(ip2)

    Dense4 = Dense(50,activation='relu')(Dense3)

    concatelayer = keras.layers.Concatenate(axis=1)([Dense2,Dense4])

    DenseL1 = Dense(228,activation='relu')(concatelayer)

    output1 = Dense(2,activation='softmax')(DenseL1)

    mainmodel = Model(inputs = [ip1,ip2],outputs = output1)                 

    mainmodel.compile('adam','categorical_crossentropy',metrics = ['accuracy'])

    print(mainmodel.summary())

    print("input shape ",mainmodel.input_shape)

    print("output shape ",mainmodel.output_shape)

    return mainmodel
model = simple_model()
hist = model.fit([fit_images,x],onehot_labels,epochs=10,batch_size=16,validation_split=0.2)
plt.figure(figsize=(10,7))

plt.subplot(1,2,1)

plt.plot(hist.history['accuracy'],label='accuracy')

plt.plot(hist.history['loss'],label='loss')

plt.legend()

plt.title("training set")

plt.grid()

plt.subplot(1,2,2)

plt.plot(hist.history['val_accuracy'],label='val_accuracy')

plt.plot(hist.history['val_loss'],label='val_loss')

plt.legend()

plt.title("validation set")

plt.grid()

plt.ylim((0,4))