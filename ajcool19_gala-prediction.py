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
import tensorflow as tf

import keras

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

%matplotlib inline
#os.chdir("/kaggle/input/dataset/")
trainlabel=pd.read_csv("/kaggle/input/dataset/train.csv")
trainlabel.head()
trainlabel.head()

trainlabel.info()

trainlabel1=trainlabel[:4501]

trainlabel2=trainlabel[4501:]
trainlabel2.head
train_image = []

train_label1=[]

for i in tqdm(range(trainlabel1.shape[0])):

    img = image.load_img('/kaggle/input/dataset/Train Images/'+trainlabel1['Image'][i],target_size=(224,224))

    label=trainlabel1['Class'][i]

    train_label1.append(label)

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

X = np.array(train_image)
X.shape
train_label1=np.array(train_label1)

#train_label1.shape
label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(train_label1)

print(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)

y=onehot_encoded
from keras.applications import VGG16



vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in vgg_conv.layers[:-4]:

    layer.trainable = False
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential



model = Sequential()

model.add(vgg_conv)

### TODO: Define your architecture.



#model.add(Conv2D(filters=20,kernel_size=2,strides=2,padding='valid',activation='relu',input_shape=(224,224,3)))

#model.add(MaxPooling2D(pool_size=2,strides=2))

#model.add(Conv2D(filters=40,kernel_size=1,strides=1,padding='valid',activation='relu'))

#model.add(Dropout(0.2))

#model.add(MaxPooling2D(pool_size=2,strides=2))

'''

model.add(Conv2D(filters=80,kernel_size=2,strides=2,padding='valid',activation='relu'))

model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Dropout(0.4))

model.add(Conv2D(filters=160,kernel_size=1,strides=1,padding='valid',activation='relu'))

model.add(MaxPooling2D(pool_size=1,strides=1))

model.add(Dropout(0.3))

model.add(Conv2D(filters=320,kernel_size=1,strides=1,padding='valid',activation='relu'))

model.add(MaxPooling2D(pool_size=1,strides=1))

model.add(Dropout(0.5))

model.add(Conv2D(filters=640,kernel_size=1,strides=1,padding='valid',activation='relu'))

model.add(Dropout(0.6))

model.add(Conv2D(filters=1280,kernel_size=1,strides=1,padding='valid',activation='relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=2,strides=2))

'''

model.add(Flatten())

model.add(Dense(800,activation='sigmoid'))

model.add(Dropout(0.2))

model.add(Dense(400,activation='sigmoid'))

model.add(Dense(200,activation='sigmoid'))

model.add(Dense(100,activation='sigmoid'))

model.add(Dense(50,activation='sigmoid'))

model.add(Dense(20,activation='sigmoid'))

model.add(Dense(4,activation='softmax'))





model.summary()
y.shape
#from keras.applications import VGG16



#vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#for layer in vgg_conv.layers[:-4]:

 #   layer.trainable = False
from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np



base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output

#x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(4, activation='sigmoid')(x)
from keras.callbacks import ModelCheckpoint  

checkpointer = ModelCheckpoint(filepath='/weights.best.from_scratch.hdf5', 

                               verbose=1, save_best_only=True)



model.compile(optimizer='Adadelta', loss='categorical_crossentropy',metrics=['accuracy','mse'])

model.fit(X,y,epochs=30,callbacks=[checkpointer])
from keras.models import Sequential

from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD
test_file=pd.read_csv("/kaggle/input/dataset/test.csv")
test_file.head()
test_file.info()
train_image=[]
test_image=[]

for i in tqdm(range(test_file.shape[0])):

    img = image.load_img('/kaggle/input/dataset/Test Images/'+test_file['Image'][i],target_size=(224,224))

    #label=trainlabel1['Class'][i]

    #train_label1.append(label)

    img = image.img_to_array(img)

    img = img/255

    test_image.append(img)
test_image=np.array(test_image)
op=[]
pred=model.predict(test_image)
print(pred[0])
test_image=[]
index=0

while index<20:

    print(train_label1[index],y[index])

    index=index+1
count=0

lb=""

for i in pred:

    x=max(i)

    z=list(i)

    pos=z.index(x)

    if pos==0:

        lb="Attire"

    if pos==1:

        lb="Decorationandsignage"

    if pos==2:

        lb="Food"

    if pos==3:

        lb="misc"

    op.append(lb)

    count=count+1

print(count)
print(op[2])
im=[]

id=0

for i in test_file["Image"]:

    im.append(i)

    id=id+1

print(id)
data = {'Image':im,'Class':op}
df = pd.DataFrame(data)
df
df.to_csv("/out.csv",index=False)