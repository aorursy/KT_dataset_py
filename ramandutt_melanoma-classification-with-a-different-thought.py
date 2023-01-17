!pip install efficientnet
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.image import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow_addons as tfa 

import matplotlib.pyplot as plt
import efficientnet.tfkeras as efn

import numpy as np
import pandas as pd
import os
import math

import wandb
#Some Pre-processing
base_path='../input/siim-isic-melanoma-classification/'
train = pd.read_csv(os.path.join(base_path, 'train.csv'))
test = pd.read_csv(os.path.join(base_path, 'test.csv'))
sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
train_img_path = base_path+'jpeg/train/'
test_img_path = base_path+'jpeg/test/'

EPOCHS=30
BATCH_SIZE=10
input_shape=(512, 512, 3)
lr=1e-3
train.head()
#Basic Helper Functions

def get_model(shape, weights):
    
   
    input = Input(shape=shape)

    base_model = efn.EfficientNetB3(weights=weights,include_top=False, input_shape=shape)
    base_model.trainable = True
    
    output = base_model(input)
    output = GlobalMaxPooling2D()(output)
    output = Dense(256)(output)
    output = LeakyReLU(alpha = 0.25)(output)
    output = Dropout(0.25)(output)

    output = Dense(16,activation="relu")(output)
    output = Dropout(0.15)(output)

    output = Dense(1,activation="sigmoid")(output)
    
    model = Model(input,output)
    
    return model
    
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        pt_1 = K.clip(pt_1, 1e-3, .999)
        pt_0 = K.clip(pt_0, 1e-3, .999)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.75):
  
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr

        progress = float(epoch - num_warmup_steps ) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

lr_schedule= get_cosine_schedule_with_warmup(lr=0.00004,num_warmup_steps=4, num_training_steps=EPOCHS)

def lrfn2(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
train.columns = [ 'img_name', 'id', 'sex', 'age', 'location', 'diagnosis',
    'benign_malignant', 'target'
]
test.columns = ['img_name', 'id', 'sex', 'age', 'location']
print(train['location'].value_counts())
print("Number of NA values in location", train['location'].isna().sum())
for df in [train, test]:
    df['location'].fillna('unknown', inplace=True) #Replacing
# Filling age and sex with appropriate values.

train['sex'].fillna(train['sex'].mode()[0], inplace=True)
train['age'].fillna(train['age'].median(), inplace=True)
print(
    f'Train missing value count: {train.isnull().sum().sum()}\nTest missing value count: {train.isnull().sum().sum()}'
)
train['location'].value_counts()
location='head/neck'    #Containing images with locations as 'Head/Neck'
train_neck_head = train.loc[train['location']==location]
train_neck_head.head()
'''
Appending '.jpg' in front of each image name so that it can be used by our Train Generator
'''

def append_ext(fn):
    return fn+".jpg"

train_neck_head["img_name"]=train_neck_head["img_name"].apply(append_ext)
train_neck_head["target"] = train_neck_head['target'].astype("str")
train_neck_head.head()
aug = ImageDataGenerator(width_shift_range=0.5,
    height_shift_range=0.5, shear_range=0.5, zoom_range=0.5,
    channel_shift_range=0.5, rescale=1/255, validation_split=0.25)

image_gen = aug.flow_from_dataframe(dataframe=train_neck_head, directory=train_img_path, x_col="img_name",
y_col="target", subset="training",
batch_size=BATCH_SIZE,
seed=42,
shuffle=True,
class_mode="binary",
target_size=(300,300))

image_gen.class_indices
model_head_neck = get_model(input_shape, 'imagenet')  #Can also use noisy-student weights
model_head_neck.summary()
#Compile the Model
'''
Using Focal loss due to imbalance
'''

model_head_neck.compile(
        optimizer='adam',
        loss = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO),
        metrics=['accuracy',tf.keras.metrics.AUC()]
    )
filepath = 'EffNetB3-Head-Neck.h5'
mc = tf.keras.callbacks.ModelCheckpoint(filepath=filepath , monitor='loss', save_weights_only=False, save_model=True, save_best_only=True)
#lr_callback2 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

callbacks = [mc, lr_schedule]
history = model_head_neck.fit(image_gen,
          epochs=2,    #Increase the Number of Epochs
          batch_size=BATCH_SIZE,          
          verbose=1,
          steps_per_epoch=math.ceil(len(train_neck_head)//BATCH_SIZE),
          callbacks=callbacks)
location='lower extremity'
train_lower_ex = train.loc[train['location']==location]
NUM_SAMPLES = len(train_lower_ex)
train_lower_ex.head()
def append_ext(fn):
    return fn+".jpg"

train_lower_ex["img_name"]=train_lower_ex["img_name"].apply(append_ext)
train_lower_ex["target"] = train_lower_ex['target'].astype("str")
train_lower_ex.head()
aug = ImageDataGenerator(width_shift_range=0.5,
    height_shift_range=0.5, shear_range=0.5, zoom_range=0.5,
    channel_shift_range=0.5, rescale=1/255, validation_split=0.25)

image_gen = aug.flow_from_dataframe(dataframe=train_lower_ex, directory=train_img_path, x_col="img_name",
y_col="target", subset="training",
batch_size=BATCH_SIZE,
seed=42,
shuffle=True,
class_mode="binary",
target_size=(300,300))

image_gen.class_indices
model_lower_ex = get_model(input_shape, 'imagenet')  #Can also use noisy-student weights
model_lower_ex.summary()
#Compile the Model
model_lower_ex.compile(
        optimizer='adam',
        loss = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO),
        metrics=['accuracy',tf.keras.metrics.AUC()]
    )
filepath = 'EffNetB3-Lower-extremity.h5'
mc = tf.keras.callbacks.ModelCheckpoint(filepath=filepath , monitor='loss', save_weights_only=False, save_model=True, save_best_only=True)
lr_callback2 = tf.keras.callbacks.LearningRateScheduler(lrfn2, verbose = True)

callbacks = [mc, lr_schedule]
history = model_lower_ex.fit(image_gen,
          epochs=2,
          batch_size=BATCH_SIZE,          
          verbose=1,
          steps_per_epoch=math.ceil(NUM_SAMPLES//BATCH_SIZE),
          callbacks=callbacks)
from PIL import Image
#from tensorflow.keras.preprocessing.image import img_to_array

def prepare_img(path, name, target_size):
    try:
        img = Image.open(path+name)
    except:
        return "File not found"
        
    img_arr = img_to_array(img)
    img_arr = resize(img_arr, target_size)
    img_arr = img_arr/255
    img_arr = np.expand_dims(img_arr, axis=0)
    
    return img_arr
location='head/neck'
test_head_neck = test.loc[test['location']==location]
print(len(test_head_neck))
test_head_neck.head()
#Same Procedure

def append_ext(fn):
    return fn+".jpg"

test_head_neck["img_name"]=test_head_neck["img_name"].apply(append_ext)
test_head_neck.head()
model_head_neck = load_model('EffNetB3-Head-Neck.h5')
preds_head=[]
img_name = []
import random
for img in list(test_head_neck['img_name']):
    img_name.append(img.split('.')[0])
    img = prepare_img(test_img_path, img, (512, 512))
    preds_head.append(model_head_neck.predict(img)[0][0])
    #preds_head.append(random.choice([0,1]))
    
len(preds_head)
df_preds_head = pd.DataFrame(list(zip(img_name, preds_head)), columns=['image_name', 'target'])
df_preds_head.head()
df_preds_head.to_csv(sub_base_path+'preds-head-neck.csv', index=False)
# lower extremity
location='lower extremity'
test_lower = test.loc[test['location']==location]
test_lower.head()
def append_ext(fn):
    return fn+".jpg"

test_lower["img_name"]=test_lower["img_name"].apply(append_ext)
test_lower.head()
model_lower = load_model('EffNetB3-Lower-extremity.h5')
preds_lower=[]
img_name = []
for img in list(test_lower['img_name']):
    img_name.append(img.split('.')[0])
    img = prepare_img(test_img_path, img, (512, 512))
    preds_lower.append(model_lower.predict(img)[0][0])
    #preds_head.append(random.choice([0,1]))
    
len(preds_lower)
df_preds_lower = pd.DataFrame(list(zip(img_name, preds_lower)), columns=['image_name', 'target'])
df_preds_lower.head()
df_preds_lower.to_csv(sub_base_path+'preds_lower.csv', index=False)
df_preds_combined = df_preds_upper.append([df_preds_unknown, df_preds_torso, df_preds_palms,
                           df_preds_oral, df_preds_lower, df_preds_head], ignore_index=True)
print(len(df_preds_combined))
df_preds_combined.head()
del sample['target']
sample = sample.merge(df_preds_combined, on='image_name')
sample.to_csv('Submissions/individual1.csv', index=False)
sample.head()
