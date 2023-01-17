import warnings

import cv2

import keras
import keras.backend as K

from keras import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU
from keras.layers import BatchNormalization, Activation, Conv2D 
from keras.layers import GlobalAveragePooling2D, Lambda
from keras.optimizers import Adam, RMSprop

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import tensorflow as tf

%matplotlib inline

print('Keras version:', keras.__version__)

warnings.simplefilter('default')
train_info = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
# Sample file path: ../input/landmark-retrieval-2020/train/0/0/0/0000059611c7d079.jpg
prefix = '../input/landmark-retrieval-2020/train/'
train_info['file_path'] = train_info['id'].apply(lambda x:prefix+ x[0]+'/'+x[1]+'/'+x[2]+'/'+x+".jpg")
train_image_files = train_info['file_path'].values

# Sample visualization
testimg = cv2.cvtColor(cv2.imread(np.random.choice(train_image_files)), cv2.COLOR_BGR2RGB)
plt.imshow(testimg)
testimg.shape

label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=True)

train_info['label'] = label_encoder.fit_transform(train_info['landmark_id'].values)
train_info['one_hot'] = one_hot_encoder.fit_transform(
                    train_info['label'].values.reshape(-1, 1))
n_cat =81313

batch_size = 48
batch_size_predict = 128
input_shape = (299,299)
def load_images(info, input_shape = input_shape):
    input_shape = tuple(input_shape)
    imgs = np.zeros((len(info), input_shape[0], input_shape[1], 3))

    for i in range(len(info)):
        fname = train_info.iloc[i]['file_path']
        try:
            img = cv2.cvtColor(
                  cv2.resize(cv2.imread(fname),input_shape),
                  cv2.COLOR_BGR2RGB)
        except:
            warnings.warn('Warning: could not read image: '+ fname +
                          '. Use black img instead.')
            img = np.zeros((input_shape[0], input_shape[1], 3))
        imgs[i,:,:,:] = img
    
    return imgs

def load_cropped_images(info, crop_p=0.2, crop='random'):
    new_res = np.array([int(input_shape[0]*(1+crop_p)), int(input_shape[1]*(1+crop_p))])
    if crop == 'random':
        cx0 = np.random.randint(new_res[0] - input_shape[0], size=len(info))
        cy0 = np.random.randint(new_res[1] - input_shape[1], size=len(info))
    else:
        if crop == 'central':
            cx0, cy0 = (new_res - input_shape) // 2                
        if crop == 'upper left':
            cx0, cy0 = 0, 0
        if crop == 'upper right':
            cx0, cy0 = new_res[1] - input_shape[1], 0
        if crop == 'lower left':
            cx0, cy0 = 0, new_res[0] - input_shape[0]
        if crop=='lower right':
            cx0, cy0 = new_res - input_shape        
        cx0 = np.repeat(np.expand_dims(cx0, 0), len(info))
        cy0 = np.repeat(np.expand_dims(cy0, 0), len(info))

    cx1 = cx0 + input_shape[0]
    cy1 = cy0 + input_shape[1]
    
    raw_imgs = load_images(info, input_shape=tuple(new_res))
    
    cropped_imgs = np.zeros((len(info), input_shape[0], input_shape[1], 3))
    for ind in range(len(info)):
        cropped_imgs[ind,:,:,:] = raw_imgs[ind,
                                           cy0[ind]:cy1[ind],
                                           cx0[ind]:cx1[ind], :]
    
    return cropped_imgs
def get_image_gen(info_arg, 
                  shuffle=True, 
                  image_aug=True, 
                  eq_dist=False, 
                  n_ref_imgs=16, 
                  crop_prob=0.5, 
                  crop_p=0.5):
    if image_aug:
        datagen = ImageDataGenerator(
            rotation_range=4.,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.5,
            channel_shift_range=25,
            horizontal_flip=True,
            fill_mode='nearest')
        
        if crop_prob > 0:
            datagen_crop = ImageDataGenerator(
                rotation_range=4.,
                shear_range=0.2,
                zoom_range=0.1,
                channel_shift_range=20,
                horizontal_flip=True,
                fill_mode='nearest')
        
    count = len(info_arg)
    while True:
        if eq_dist:
            def sample(df):
                return df.sample(min(n_ref_imgs, len(df)))
            info = info_arg.groupby('landmark_id', group_keys=False).apply(sample)
        else:
            info = info_arg
        print('Generate', len(info), 'for the next round.')
        
        #shuffle data
        if shuffle and count >= len(info):
            info = info.sample(frac=1)
            count = 0
            
        # load images
        for ind in range(0,len(info), batch_size):
            count += batch_size

            y = info['landmark_id'].values[ind:(ind+batch_size)]
            
            if np.random.rand() < crop_prob:
                imgs = load_cropped_images(info.iloc[ind:(ind+batch_size)], 
                                           crop_p=crop_p*np.random.rand() + 0.01, 
                                           crop='random')
                if image_aug:
                    cflow = datagen_crop.flow(imgs, 
                                              y, 
                                              batch_size=imgs.shape[0], 
                                              shuffle=False)
                    imgs, y = next(cflow)                    
            else:
                imgs = load_images(info.iloc[ind:(ind+batch_size)])
                if image_aug:
                    cflow = datagen.flow(imgs, 
                                       y, 
                                       batch_size=imgs.shape[0], 
                                       shuffle=False)
                    imgs, y = next(cflow)             

            imgs = preprocess_input(imgs)
    
            y_l = label_encoder.transform(y[y>=0.])        
            y_oh = np.zeros((len(y), n_cat))
            y_oh[y >= 0., :] = one_hot_encoder.transform(y_l.reshape(-1,1)).todense()
                    
            yield imgs, y_oh
            
train_gen = get_image_gen(train_info, 
                          eq_dist=False, 
                          n_ref_imgs=256, 
                          crop_prob=0.5, 
                          crop_p=0.5)
X_example, y_example = next(train_gen)
plt.imshow(X_example[1,:,:,:]/2. + 0.5)
K.clear_session()

x_model = Xception(input_shape=list(input_shape) + [3], 
                   weights='imagenet', 
                   include_top=False)
x_model.summary()

print((x_model.layers[85]).name)
print((x_model.layers[25]).name)
print((x_model.layers[15]).name)
for layer in x_model.layers:
    layer.trainable = True

for layer in x_model.layers[:85]:
    layer.trainable = False   
    
x_model.summary()
gm_exp = tf.Variable(3., dtype=tf.float32)
def generalized_mean_pool_2d(X):
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 
                           axis=[1,2], 
                           keepdims=False)+1.e-8)**(1./gm_exp)
    return pool
X_feat = Input(x_model.output_shape[1:])

lambda_layer = Lambda(generalized_mean_pool_2d)
lambda_layer.trainable_weights.extend([gm_exp])
X = lambda_layer(X_feat)
X = Dropout(0.05)(X)
X = Activation('relu')(X)
X = Dense(n_cat, activation='softmax')(X)

top_model = Model(inputs=X_feat, outputs=X)
top_model.summary()
X_image = Input(list(input_shape) + [3])

X_f = x_model(X_image)
X_f = top_model(X_f)

model = Model(inputs=X_image, outputs=X_f)
model.summary()
def get_custom_loss(rank_weight=1., epsilon=1.e-9):
    def custom_loss(y_t, y_p):
        losses = tf.reduce_sum(-y_t*tf.math.log(y_p+epsilon) - (1.-y_t)*tf.math.log(1.-y_p+epsilon), 
                               axis=-1)
        
        pred_idx = tf.argmax(y_p, axis=-1)
        
        mask = tf.one_hot(pred_idx, 
                          depth=y_p.shape[1], 
                          dtype=tf.bool, 
                          on_value=True, 
                          off_value=False)
        pred_cat = tf.boolean_mask(y_p, mask)
        y_t_cat = tf.boolean_mask(y_t, mask)
        
        n_pred = tf.shape(pred_cat)[0]
        _, ranks = tf.nn.top_k(pred_cat, k=n_pred)
        
        ranks = tf.cast(n_pred-ranks, tf.float32)/tf.cast(n_pred, tf.float32)*rank_weight
        rank_losses = ranks*(-y_t_cat*tf.math.log(pred_cat+epsilon)
                             -(1.-y_t_cat)*tf.math.log(1.-pred_cat+epsilon))        
        
        return rank_losses + losses
    return custom_loss
def batch_GAP(y_t, y_p):
    pred_cat = tf.argmax(y_p, axis=-1)    
    y_t_cat = tf.argmax(y_t, axis=-1) * tf.cast(
        tf.reduce_sum(y_t, axis=-1), tf.int64)
    
    n_pred = tf.shape(pred_cat)[0]
    is_c = tf.cast(tf.equal(pred_cat, y_t_cat), tf.float32)

    GAP = tf.reduce_mean(
          tf.cumsum(is_c) * is_c / tf.cast(
              tf.range(1, n_pred + 1), 
              dtype=tf.float32))
    
    return GAP

def binary_crossentropy_n_cat(y_t, y_p):
    return keras.metrics.binary_crossentropy(y_t, y_p) * n_cat
opt = Adam(lr=0.0001)
loss = get_custom_loss(1.0)
model.compile(loss=loss, 
              optimizer=opt, 
              metrics=[binary_crossentropy_n_cat, 'accuracy', batch_GAP])
checkpoint1 = ModelCheckpoint('dd_checkpoint-1.h5', 
                              period=1, 
                              verbose=1, 
                              save_weights_only=True)
checkpoint2 = ModelCheckpoint('dd_checkpoint-2.h5', 
                              period=1, 
                              verbose=1, 
                              save_weights_only=True)
checkpoint3 = ModelCheckpoint('dd_checkpoint-3-best.h5', 
                              period=1, 
                              verbose=1, 
                              monitor='loss', 
                              save_best_only=True, 
                              save_weights_only=True)
model.fit_generator(train_gen, 
                    steps_per_epoch=len(train_info) / batch_size / 8, 
                    verbose=2,
                    epochs=100)

# model.fit_generator(train_gen, 
#                     steps_per_epoch=len(train_info) / batch_size / 8, 
#                     epochs=1, 
#                     callbacks=[checkpoint1, checkpoint2, checkpoint3])
# Doesn't work right now

tf.saved_model.save(model, "../input/output/", signatures=None, options=None)
# Code for converting to zip file

from zipfile import ZipFile

with ZipFile('submission.zip','w') as zip:           
    zip.write('../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model/saved_model.pb', arcname='saved_model.pb') 
    zip.write('../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model/variables/variables.data-00000-of-00001', arcname='variables/variables.data-00000-of-00001') 
    zip.write('../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model/variables/variables.index', arcname='variables/variables.index') 