from IPython.display import Image
Image(filename='../input/landmark-retrieval-2020/train/0/0/0/000014b1f770f640.jpg')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import tensorflow as tf
import cv2
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU, LeakyReLU,Reshape, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.applications import ResNet101
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


from tensorflow.keras import activations
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Input, Reshape, ZeroPadding2D
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/landmark-retrieval-2020/train.csv")
def get_paths(sub):
    index = ["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"]

    paths = []

    for a in index:
        for b in index:
            for c in index:
                try:
                    paths.extend([f"../input/landmark-retrieval-2020/{sub}/{a}/{b}/{c}/" + x for x in os.listdir(f"../input/landmark-retrieval-2020/{sub}/{a}/{b}/{c}")])
                except:
                    pass

    return paths

train_path = train
train_path["id"] = train_path.id.map(lambda path: f"../input/landmark-retrieval-2020/train/{path[0]}/{path[1]}/{path[2]}/{path}.jpg")
##Old implementation - changed after suggestion from @nawidsayed
'''
train_path = train

rows = []
for i in tqdm(range(len(train))):
    row = train.iloc[i]
    path  = list(row["id"])[:3]
    temp = row["id"]
    row["id"] = f"../input/landmark-retrieval-2020/train/{path[0]}/{path[1]}/{path[2]}/{temp}.jpg"
    rows.append(row["id"])
    
rows = pd.DataFrame(rows)
train_path["id"] = rows
'''
batch_size = 128
# seed = 42
shape = (64, 64, 3) ##desired shape of the image for resizing purposes
val_sample = 0.1 # 10 % as validation sample
train_labels = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
train_labels.head()
k =train[['id','landmark_id']].groupby(['landmark_id']).agg({'id':'count'})
k.rename(columns={'id':'Count_class'}, inplace=True)
k.reset_index(level=(0), inplace=True)
freq_ct_df = pd.DataFrame(k)
freq_ct_df.head()
train_labels = pd.merge(train,freq_ct_df, on = ['landmark_id'], how='left')
train_labels.head()
freq_ct_df.sort_values(by=['Count_class'],ascending=False,inplace=True)
freq_ct_df.head()
freq_ct_df_top100 = freq_ct_df.iloc[:100]
top100_class = freq_ct_df_top100['landmark_id'].tolist()
top100class_train = train_path[train_path['landmark_id'].isin (top100_class) ]
top100class_train.shape
def getTrainParams():
    data = top100class_train.copy()
    le = preprocessing.LabelEncoder()
    data['label'] = le.fit_transform(data['landmark_id'])
    lbls = top100class_train['landmark_id'].tolist()
    lb = LabelBinarizer()
    labels = lb.fit_transform(lbls)
    
    return np.array(top100class_train['id'].tolist()),np.array(labels),le
class Landmark2020_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, shape, shuffle = False, use_cache = False, augment = False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        y = X
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = X/255.0
        X = X/255.0
#         window_name = 'image'

#         print('X:', X)
#         print('y', y)
                
#         if self.augment == True:
#             seq = iaa.Sequential([
#                 iaa.OneOf([
#                     iaa.Fliplr(0.5), # horizontal flips
                    
#                     iaa.ContrastNormalization((0.75, 1.5)),
#                     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#                     iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    
#                     iaa.Affine(rotate=0),
#                     #iaa.Affine(rotate=90),
#                     #iaa.Affine(rotate=180),
#                     #iaa.Affine(rotate=270),
#                     iaa.Fliplr(0.5),
#                     #iaa.Flipud(0.5),
#                 ])], random_order=True)

#             X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
#             y = np.concatenate((y, y, y, y), 0)
        
        return X, y
    
    def on_epoch_end(self):
        
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        im = np.array(Image.open(str(path)).convert('RGB').resize((self.shape[0], self.shape[1])))
#         print(im)
        return im
nlabls = top100class_train['landmark_id'].nunique()
paths, labels,_ = getTrainParams()
keys = np.arange(paths.shape[0], dtype=np.int)  
# np.random.seed(seed)
np.random.shuffle(keys)
lastTrainIndex = int((1-val_sample) * paths.shape[0])

pathsTrain = paths[0:lastTrainIndex]
labelsTrain = labels[0:lastTrainIndex]

pathsVal = paths[lastTrainIndex:]
labelsVal = labels[lastTrainIndex:]

print(paths.shape, labels.shape)
print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)
train_generator = Landmark2020_DataGenerator(pathsTrain, labelsTrain, batch_size, shape, use_cache=False, augment = False, shuffle = True)
val_generator = Landmark2020_DataGenerator(pathsVal, labelsVal, batch_size, shape, use_cache=False, shuffle = False)
def dres_identity(x, filters): 
    # resnet block where dimension doesnot change.
    # The skip connection is just simple identity conncection
    # There will be 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters

    # first block 
    x = Conv2DTranspose(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)


    # second block # bottleneck (but size kept same with padding)
    x = Conv2DTranspose(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = Conv2DTranspose(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    # add the input 
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x
def dres_conv(x, s, filters):
    # here the input size changes
    x_skip = x
    f1, f2 = filters

    # third block
    x = Conv2DTranspose(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block
    x = Conv2DTranspose(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block
    x = Conv2DTranspose(f1, kernel_size=(1, 1), strides=(s, s), padding='valid')(x)
    # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)

    # shortcut 
    x_skip = Conv2DTranspose(f1, kernel_size=(1, 1), strides=(s, s), padding='valid')(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add 
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x
input_im = Input(shape=(64, 64, 3))
Encoder = ResNet101(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
x = Encoder(input_im)
x = Flatten()(x)
encoding = Dense(2048, kernel_initializer='he_normal')(x)
encoder = tf.keras.Model(inputs=input_im, outputs=encoding, name='Encoder')
# Decoder
dec_input = Input(shape=(2048,))
x = Dense(2 * 2 * 2048, kernel_initializer='he_normal')(dec_input)
x = Reshape((2, 2, 2048))(x)

x = dres_conv(x, s=2, filters=(512, 2048))
x = dres_identity(x, filters=(512, 2048))
x = dres_identity(x, filters=(512, 2048))

x = dres_conv(x, s=2, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))
x = dres_identity(x, filters=(256, 1024))


x = dres_conv(x, s=2, filters=(128, 512))
x = dres_identity(x, filters=(128, 512))
x = dres_identity(x, filters=(128, 512))
x = dres_identity(x, filters=(128, 512))

x = dres_conv(x, s=2, filters=(64, 256))
x = dres_identity(x, filters=(64, 256))
x = dres_identity(x, filters=(64, 256))
x = Conv2DTranspose(3, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
# decoded = Activation(activations.relu)(x)
decoded = Activation(activations.sigmoid)(x)
decoder = tf.keras.Model(inputs=dec_input, outputs=decoded, name='Decoder')
enc_input = Input(shape=(64, 64, 3))
encoding = encoder(enc_input)
decoded = decoder(encoding)
auto_encoder = tf.keras.Model(inputs=enc_input, outputs=decoded, name='AutoEncoder')
auto_encoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
auto_encoder.summary()
# epochs = 2
# use_multiprocessing = True 
#workers = 1
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
print(tf.__version__)
base_cnn = auto_encoder.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=64,
    #class_weight = class_weights,
    epochs=5,
    #callbacks = [clr],
    use_multiprocessing=True,
    #workers=workers,
    verbose=1)
auto_encoder.save_weights('Auto_Encoder.h5')
encoder.save_weights('Encoder.h5')
decoder.save_weights('Decoder.h5')
from IPython.display import FileLink
FileLink(r'Encoder.h5')
FileLink(r'Decoder.h5')
FileLink(r'Auto_Encoder.h5')