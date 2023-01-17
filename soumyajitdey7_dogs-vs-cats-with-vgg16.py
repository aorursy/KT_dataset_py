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
import keras, os, cv2, random
from keras.models import Sequential # using squential for creating sequential model
#sequential model means all the layers of the model will be arranged in sequence
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator, load_img
#ImageDataGenerator help label the data so that it can easily import data into the model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns
from tqdm import tqdm
from random import shuffle 
import os, cv2 ,random
import os
for dirname, _, filenames in os.walk('/kaggle/working/train'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import numpy as np
import pandas as pd
import zipfile, os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip:
    zip.extractall()    
    zip.close()
sample_sub = pd.read_csv('/kaggle/input/dogs-vs-cats/sampleSubmission.csv')
print(sample_sub.head())

sample_img = load_img('/kaggle/working/train/cat.6562.jpg') # cute pic :)
plt.imshow(sample_img)
filenames = os.listdir('/kaggle/working/train')

labels = []
for filename in filenames:
    label = filename.split('.')[0] # splits on the first dot
    if label == 'cat':
        labels.append('0')
    else:
        labels.append('1')
        
df = pd.DataFrame({'id': filenames, 'label':labels })
print(df.shape)
df.head()
        
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

import os
print(os.listdir("../input"))


Test_Size= 0.5
Random_State = 2018
Batch_Size = 64
No_Epochs = 20
Num_Classes = 2
Sample_Size  = 20000
PATH = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'
TRAIN_FOLDER = './train/'
TEST_FOLDER =  './test1'
IMG_SIZE = 224
RESNET_WEIGHTS_PATH = '/kaggle/input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#Read The Data

train_image_path= os.path.join(PATH, "/kaggle/input/dogs-vs-cats/train.zip")
test_image_path= os.path.join(PATH, "/kaggle/input/dogs-vs-cats/test1.zip")
import zipfile
with zipfile.ZipFile(train_image_path,"r") as z:
    z.extractall(".")
with zipfile.ZipFile(test_image_path,"r") as z:
    z.extractall(".")
train_image_list = os.listdir("./train")[0:Sample_Size]
test_image_list = os.listdir("./test1")
def label_pet_image_one_hot_encoder(img):
    pet = img.split('.')[-3]
    if pet == 'cat': return [1,0]
    elif pet == 'dog' : return [0,1]
def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(DATA_FOLDER,img)
        if(isTrain):
            label = label_pet_image_one_hot_encoder(img)
        else:
            label = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data_df.append([np.array(img),np.array(label)])
    shuffle(data_df)
    return data_df
def plot_image_list_count(data_image_list):
    labels= []
    for img in data_image_list:
        labels.append(img.split('.')[-3])
    sns.countplot(labels)
    plt.title('Cats and Dogs')
    
plot_image_list_count(train_image_list)
plot_image_list_count(os.listdir(TRAIN_FOLDER))
train = process_data(train_image_list, TRAIN_FOLDER)
def show_images(data, isTest=False):
    f, ax = plt.subplots(5,5, figsize=(15,15))
    for i, data in enumerate(data[:25]):
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)
        if label == 1:
            str_label= 'Dog'
        elif label == 0:
            str_label = 'Cat'
        if(isTest):
            str_label= "None"
        ax[i//5, i%5].imshow(img_data)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title("Label: {}".format(str_label))
    plt.show()

show_images(train)
test = process_data(test_image_list, TEST_FOLDER, False)
show_images(test, True)
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array([i[1] for i in train])
from IPython.display import SVG
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Flatten, Input, Lambda
from glob import glob
%matplotlib inline
IMAGE_SHAPE=[224, 224]
#add preprocessing layer to the front of vgg
vgg16 = VGG16(input_shape= IMAGE_SHAPE + [3], weights = 'imagenet', include_top= False)
for layer in vgg16.layers:
    layer.trainable = False
#Useful for getting the number of classes
folders = glob('./train')
#Making the Flatten layer
X = Flatten()(vgg16.output)
prediction = Dense(len(folders), activation= 'softmax')(X)
model= Model(inputs= vgg16.input, outputs= prediction)
#view model structure
model.summary()
#tell the model which cost and optimization method to use
model.compile(
     loss= 'categorical crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)
conv_base = VGG16(weights='imagenet', include_top =False, input_shape=(200,200,3))

#include_top refers to including the Dense layer on top of the network (1000 classes, in this case)

model = models.Sequential()
model .add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

# freezing the convolutional base so that its weights aren't updated:
#conv_base.trainable = False
# only the weights of the Dense layers will be updated

# we're gonna do some fine-tuning by training a part of the convolutional base
# it's basically freezing all the layers except the most abstract ones

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])
model.summary()
#use the image data generator to import images from dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(rescale = 1./255,
                                 shear_range= 0.2,
                                 zoom_range = 0.2,
                                horizontal_flip= True)
test_datagen= ImageDataGenerator(rescale= 1./255)
train_path = '/kaggle/working/train'
train_df, validation_df = train_test_split(df, test_size=0.1)

train_size = train_df.shape[0]
validation_size = validation_df.shape[0]
batch_size = 20

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    '/kaggle/working/train/',
                                                    x_col='id',
                                                    y_col='label',
                                                    class_mode='binary',
                                                   target_size=(200,200),
                                                   batch_size=batch_size)

validation_generator = test_datagen.flow_from_dataframe(validation_df,
                                                       '/kaggle/working/train/',
                                                       x_col='id',
                                                       y_col='label',
                                                       class_mode='binary',
                                                       target_size=(200,200),
                                                       batch_size=batch_size)

history = model.fit_generator(train_generator,
                             steps_per_epoch=train_size//batch_size,
                             epochs=5,
                             validation_data=validation_generator,
                             validation_steps=validation_size//batch_size)

model.save('catsvsdogs_vgg16.h5')

