# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               
from keras.preprocessing import image                  
from tqdm import tqdm

CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
NUM_CATEGORIES = len(CATEGORIES)

data_dir = '../input/plant-seedlings-classification/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
sample_submission = pd.read_csv('../input/plant-seedlings-classification/sample_submission.csv')
sample_submission.head(5)
for category in CATEGORIES:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))
train = []
for category_id, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(train_dir, category)):
        train.append(['train/{}/{}'.format(category, file), category_id, category])
train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])
train.shape
train.head(5)
import seaborn as sns
sns.countplot(x='category',data=train).set_title('distribution of different category');
plt.xticks(rotation=90);
test = []
for file in os.listdir(test_dir):
    test.append(['test/{}'.format(file), file])
test = pd.DataFrame(test, columns=['filepath', 'file'])

test.shape
test.head(10)
def read_img(filepath, size):
    img = image.load_img(os.path.join(data_dir, filepath), target_size=size)
    img = image.img_to_array(img)
    img = np.array(img)
    return img
x_train=[]
for file in tqdm(train['file']):
    img = read_img(file, (224, 224))
    x_train.append(img)

    
    
x_train=np.array(x_train)
x_train.shape
x_test=[]
for file in tqdm(test['filepath']):
    img=read_img(file,(224,224))
    x_test.append(img)
x_test=np.array(x_test)

x_test.shape
y_train=train['category_id']
y_train.shape
from sklearn.preprocessing import LabelBinarizer 
label_binarizer = LabelBinarizer()
label_binarizer.fit(y_train)

y_train=label_binarizer.transform(y_train)
y_train.shape
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(x_train,y_train,test_size=0.05, random_state=42)
X_train=X_train/255
X_valid=X_valid/255
x_test=x_test/255
from keras.preprocessing.image import ImageDataGenerator
datagen_train = ImageDataGenerator(
    width_shift_range=0.2,  # randomly shift images horizontally 
    height_shift_range=0.2,# randomly shift images vertically 
    
    horizontal_flip=True) # randomly flip images horizontally

# fit augmented image generator on data
datagen_train.fit(X_train)
from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model
from keras.callbacks import ModelCheckpoint
!ls ../input/keras-pretrained-models/
cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
!cp ../input/keras-pretrained-models/vgg* ~/.keras/models/
!ls ~/.keras/models
from keras import applications
model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))
model.summary()
for layer in model.layers:
    layer.trainable = False
model.summary()
x = model.output



x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
predictions = Dense(12, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy",  optimizer ='adam', metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath='vgg16.hdf5', verbose=1, save_best_only=True)
'''
model_final.fit_generator(datagen_train.flow(X_train, Y_train, batch_size=16), validation_data=(X_valid, Y_valid),
                         epochs=2,steps_per_epoch=X_train.shape[0],callbacks=[checkpointer], verbose=1)
'''
