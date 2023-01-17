# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

import seaborn as sns

import zipfile

import io

import os

import keras

from keras.models import Model, Sequential

from keras.utils import layer_utils

from keras.preprocessing import image

from keras.applications.vgg16 import VGG16,preprocess_input, decode_predictions



# Any results you write to the current directory are saved as output.
from mpl_toolkits.axes_grid1 import ImageGrid

from tqdm import tqdm
!ls ../input/plant-seedlings-classification/train
CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

NUM_CATEGORIES = len(CATEGORIES)
train_dir = '../input/plant-seedlings-classification/train'

test_dir = '../input/plant-seedlings-classification/test'
for category in CATEGORIES:

    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir,category)))))
!ls ../input/keras-pretrained-models
cache_dir = os.path.expanduser(os.path.join('~','.keras'))

if not os.path.exists(cache_dir):

    os.mkdir(cache_dir)

    

model_dir = os.path.join(cache_dir,'models')

if not os.path.exists(model_dir):

    os.mkdir(model_dir)

    

    
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/

!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
!ls ~/.keras/models
sample_per_category = 100

seed = 1999

data_dir = '../input/plant-seedlings-classification'

sample_submisson = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'))
sample_submisson.head()
train=[]

for category_id, category in enumerate(CATEGORIES):

    for file in os.listdir(os.path.join(train_dir,category)):

        train.append(['train/{}/{}'.format(category,file),category_id,category])

        
train = pd.DataFrame(train,columns=['file','category_id','category'])

train.head(2)
train = pd.concat(train[train['category']==c][:sample_per_category] for c in CATEGORIES)

train=train.sample(frac=1)

train.index=np.arange(len(train))
train.head(2)
train.shape
test = []

for file in os.listdir(test_dir):

    test.append(['test/{}'.format(file),file])

test=pd.DataFrame(test,columns=['file_path','file'])

test.head(2)
test.shape
def read_img(file_path,img_size):

    img = image.load_img(file_path,target_size=img_size)

    img = image.img_to_array(img)

    return img
img_sample=read_img(os.path.join(data_dir,train.loc[0,'file']),(224,224))
plt.imshow(img_sample)
fig = plt.figure(1,figsize=(NUM_CATEGORIES,NUM_CATEGORIES))

grid = ImageGrid(fig, 111, nrows_ncols=(NUM_CATEGORIES,NUM_CATEGORIES), axes_pad=0.05)

i = 0

for category_id, category in enumerate(CATEGORIES):

    for file in train[train['category']==category]['file'].values[:NUM_CATEGORIES]:

        ax=grid[i]

        img = read_img(os.path.join(data_dir,file),(224,224))

        ax.imshow(img)

        ax.axis('off')

        if i%NUM_CATEGORIES == NUM_CATEGORIES-1:

            ax.text(250,112,file.split('/')[1], verticalalignment='center')

        i += 1

plt.show()
np.random.seed(seed)
rnd = np.random.random(len(train))

train_index = rnd<0.8

valid_index=rnd>=0.8

ytrain=train.loc[train_index,'category_id'].values

yvalid=train.loc[valid_index,'category_id'].values

len(ytrain),len(yvalid)
INPUT_SIZE=224

POOLING='avg'

x_train=np.zeros((len(train),INPUT_SIZE,INPUT_SIZE,3),dtype=np.float32)

for i,file in tqdm(enumerate(train['file'])):

    img = read_img(os.path.join(data_dir,file),(INPUT_SIZE,INPUT_SIZE))

    x=preprocess_input(np.expand_dims(img.copy(),axis=0))

    x_train[i]=x

print('Train image shape: {} size: {:,}'.format(x_train.shape,x_train.size))
xtrain=x_train[train_index]

xvalid=x_train[valid_index]

print((xtrain.shape,xvalid.shape))
vgg_bottleneck = VGG16(weights='imagenet', include_top=False, pooling=POOLING)

train_vgg_bf = vgg_bottleneck.predict(xtrain, batch_size=32, verbose=1)

valid_vgg_bf = vgg_bottleneck.predict(xvalid, batch_size=32, verbose=1)

print('VGG train bottleneck features shape: {} size: {:,}'.format(train_vgg_bf.shape, train_vgg_bf.size))

print('VGG valid bottleneck features shape: {} size: {:,}'.format(valid_vgg_bf.shape, valid_vgg_bf.size))
FCN = Sequential()
train_vgg_bf.shape
one_hot_labels = keras.utils.to_categorical(ytrain, num_classes=12)
valid_labels = keras.utils.to_categorical(yvalid, num_classes=12)
FCN.add(keras.layers.Dense(input_shape=(512,),activation='relu', 

                           units=512, kernel_regularizer=keras.regularizers.l2(l=0.001)))
FCN.add(keras.layers.Dense(activation='relu', 

                           units=128, kernel_regularizer=keras.regularizers.l2(l=0.001)))
FCN.add(keras.layers.Dense(activation='softmax', 

                           units=12))
FCN.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy']) 
FCN.fit(train_vgg_bf,one_hot_labels)


score = FCN.evaluate(valid_vgg_bf, valid_labels)
FCN.metrics_names
score
x_test = np.zeros((len(test), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, file in tqdm(enumerate(test['file_path'])):

    img = read_img(os.path.join(data_dir,file), (INPUT_SIZE, INPUT_SIZE))

    x = preprocess_input(np.expand_dims(img.copy(), axis=0))

    x_test[i] = x

print('test Images shape: {} size: {:,}'.format(x_test.shape, x_test.size))
test_x_bf = vgg_bottleneck.predict(x_test, batch_size=32, verbose=1)

print('Test bottleneck features shape: {} size: {:,}'.format(test_x_bf.shape, test_x_bf.size))
test_preds = FCN.predict(test_x_bf)
np.argmax(test_preds,axis=1).shape
test_pred_one = np.argmax(test_preds,axis=1)

test['category_id'] = test_pred_one

test['species'] = [CATEGORIES[c] for c in test_pred_one]

test[['file', 'species']].to_csv('submission.csv', index=False)