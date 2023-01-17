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
import os
import numpy as np
import pandas as pd
from PIL import Image
from zipfile import ZipFile
import matplotlib.pyplot as plt
import cv2
import math
from imgaug import augmenters as iaa
%matplotlib inline
os.listdir('../input')
PATH_BASE = '../input/'
TRAIN_BASE = 'human-protein-atlas-image-classification/'
MODEL_BASE = '4-channel-v2-with-rare-case-extension-t2/'
PATH_TRAIN = PATH_BASE+TRAIN_BASE+'train/'
raw_labels = pd.read_csv(PATH_BASE+TRAIN_BASE+'train.csv')
data_names = os.listdir(PATH_TRAIN)

#extract label names and labels array[{name: ,label:}]
labels = []
for name, label in zip(raw_labels['Id'],raw_labels['Target'].str.split(" ")):
    labels.append({
        'name':name,
        'label':label
    })
#Split data to train/dev set
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(labels, test_size=0.2)
print('train: ' + str(len(train_idx)) + '\n'+ 'validation: ' + str(len(test_idx)))
y_cat_train_dic = {}
for icat in range(28):
    target = str(icat)
    y_cat_train_5 = np.array([int(target in ee['label']) for ee in train_idx])
    y_cat_train_dic[icat] = y_cat_train_5
up_sample = {}
for k in y_cat_train_dic:
    v = y_cat_train_dic[k].sum()
    up_sample[k] = np.round(v / len(train_idx), 5)
print(up_sample)
def plt_barh(x, y, title):
    fig, ax = plt.subplots(figsize=(15,7))
    width = 0.75
    ind = np.arange(len(up_sample))  # the x locations for the groups
    ax.barh(ind, y, width, color="blue")
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(x, minor=False)
    plt.title(title)
    for i, v in enumerate(y):
        ax.text(v, i , str(v), color='blue', fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('y')
x = list(up_sample.keys())
y = list(up_sample.values())
plt_barh(x, y, 'data imbalance')
#np.random.seed(18)
test = labels[10]
print(test)
print(test['name'])
print(test['label'])

fig, ax = plt.subplots(1,4,figsize=(12,12))
fig.tight_layout()
#Try different mix method
names = [n['name'] for n in np.random.choice(labels, 1)]
R = np.array(Image.open(PATH_TRAIN+names[0]+'_red.png'))
ax[0].imshow(R,cmap='Reds')
ax[0].set_title('R')
G = np.array(Image.open(PATH_TRAIN+names[0]+'_green.png'))
ax[1].imshow(G,cmap='Greens')
ax[1].set_title('G')
B = np.array(Image.open(PATH_TRAIN+names[0]+'_blue.png'))
ax[2].imshow(B,cmap='Blues')
ax[2].set_title('B')
Y = np.array(Image.open(PATH_TRAIN+names[0]+'_yellow.png'))
ax[3].imshow(Y,cmap='YlOrBr')
ax[3].set_title('Y')

BY = (B+Y)
BY[BY>255] = 255
RY = (R+Y)
RY[RY>255] = 255
GY = (G+Y)
GY[GY>255] = 255

IMG = np.stack((R, G, B) ,axis=-1)
IMG2 = np.stack((R, G, BY) ,axis=-1)
IMG3 = np.stack((RY, G, B) ,axis=-1)
IMG4 = np.stack((R, GY, B) ,axis=-1)
#IMG = np.divide(IMG, 255)
IMG = cv2.resize(IMG,(299,299))

fig2, ax2 = plt.subplots(2,2)
fig2.set_size_inches(12,12)
ax2[0,0].set_title('R,G,B')
ax2[0,0].imshow(IMG)
ax2[0,1].set_title('R,G,BY')
ax2[0,1].imshow(IMG2)
ax2[1,0].set_title('RY,G,B')
ax2[1,0].imshow(IMG3)
ax2[1,1].set_title('R,GY,B')
ax2[1,1].imshow(IMG4)
IMG.shape
#Define data_generator

class data_generator:
    
    def __init__(self):
        pass
    
    def batch_train(self, idx, batch_size, shape, augment=True):
        #extract eandom name and corresponding label
        while True:
            name_list = []
            label_list = []

            for n in np.random.choice(idx, batch_size):
                name_list.append(n['name'])
                int_label = list(map(int, n['label']))
                label_list.append(int_label)

            #batch_images = 提取images存成array, shape=(batch_size, shpae[0], shape[1], shpae[2]) = batch_images
            batch_images = np.zeros((batch_size, shape[0], shape[1], shape[2]))
            i = 0
            for name in name_list:
                image = self.load_img(name, shape)
                if augment:
                    image = self.augment(image)
                batch_images[i] = image
                i+=1

            #batch_labels = 提取labels轉換為multiple one-hot, shape=(batch_size, 28)
            batch_labels = np.zeros((batch_size, 28))
            j = 0
            for label in label_list:
                batch_labels[j][label] = 1
                j+=1

            yield batch_images, batch_labels
        
    def load_img(self, name, shape):
        R = np.array(Image.open(PATH_TRAIN+name+'_red.png'))
        G = np.array(Image.open(PATH_TRAIN+name+'_green.png'))
        B = np.array(Image.open(PATH_TRAIN+name+'_blue.png'))
        Y = np.array(Image.open(PATH_TRAIN+name+'_yellow.png'))
        image = np.stack((R, G, B, Y) ,axis=-1)
        image = cv2.resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image
    
    def augment(self, image):
        aug = iaa.OneOf([
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ])
        image = aug.augment_image(image)
        return image
from keras import applications
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
import tensorflow as tf
import keras.backend as K
K.clear_session()
SHAPE = (299,299,4)
BATCH_SIZE = 24

def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def show_history(history):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('f1')
    ax[1].plot(history.epoch, history.history["f1"], label="Train f1")
    ax[1].plot(history.epoch, history.history["val_f1"], label="Validation f1")
    ax[2].set_title('acc')
    ax[2].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[2].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    
def f1_loss(y_true, y_pred):
    K_epsilon = K.epsilon()
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K_epsilon)
    r = tp / (tp + fn + K_epsilon)

    f1 = 2*p*r / (p+r+K_epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)

def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)
        
        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise
        
        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise
        
        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """

    # Define epsilon so that the backpropagation will not result in NaN
    # for 0 divisor case
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    #y_pred = y_pred + epsilon
    # Clip the prediciton value
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
    # Calculate alpha_t
    alpha_factor = K.ones_like(y_true)*alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1-alpha_factor)
    # Calculate cross entropy
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1-p_t), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.sum(loss, axis=1)
    
    return loss

def SortedDict(adict): 
    new_dict = {}
    ks = adict.keys() 
    ks = sorted(ks)
    for key in ks:
        new_dict[key] = adict[key]
    return new_dict
# As for my examine, this focal loss should works but I'm not 100% sure, so i did a small tensor graph
# for checkng the value between function and my own math calculation.
y_true = Input(shape=(None,))
y_pred = Input(shape=(None,))
loss_function = K.Function(inputs=[y_true,y_pred], outputs=[binary_focal_loss(y_true, y_pred)])
math_loss = -0.75*math.pow((1-0.6), 2)*math.log(0.6) - 0.25*math.pow((1-0.1), 2)*math.log(0.1)
print('By manual calculate focal_loss: ', math_loss)
tensor_loss = loss_function([[[1,0,0,0,1]],[[1,0,0,0.4,0.1]]])
print('By tensor input via binary_focal loss', tensor_loss[0][0])
"""
# load base model
INPUT_SHAPE = (299,299,3)
base_model = applications.InceptionResNetV2(include_top=False ,weights='imagenet', input_shape=INPUT_SHAPE)

# Add top-model to base_model
def make_classifier_model(input_dim=(8,8,1536)):
    inp = Input(shape=input_dim)
    X = Conv2D(128, kernel_size=(3,3), activation='relu')(inp)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.25)(X)
    X = Conv2D(64, kernel_size=(1,1), activation='relu')(X)
    X = BatchNormalization()(X)
    X = Flatten()(X)  # this converts our 3D feature maps to 1D feature vectors
    X = Dense(512, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    X = Dense(256, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    X = Dense(28)(X)
    pred = Activation('sigmoid')(X)
    classifier_model = Model(inp, pred, name='classifier_model')
    return classifier_model

# Add 4-channdel input layers to base_model
def make_input_model(shape=SHAPE):
    inp = Input(shape=shape, name='input0')
    pred = Conv2D(3,kernel_size=1,strides=1,padding='same',activation='tanh',
                  kernel_regularizer=regularizers.l2(1e-4))(inp)
    input_model = Model(inp, pred, name='input_model')
    return input_model

# Create model piece
classifier_model = make_classifier_model()
input_model = make_input_model()

# Combine models
inp = Input(shape=SHAPE, name='inputs')
X = input_model(inp)
X = base_model(X)
pred = classifier_model(X)
model = Model(inp, pred, name='full_model')

model.summary()
"""
#Use this cell to read model & weight
model = load_model(PATH_BASE+MODEL_BASE+'fine_tune_weights.hdf5', custom_objects={'f1':f1, 'binary_focal_loss': binary_focal_loss})
for layer in model.layers[:]:
    layer.trainable =True

model.compile(
    loss=[binary_focal_loss],  
    optimizer=Adam(1e-4),
    metrics=['acc', f1])

#verbose = 2 for every epoch output log
checkpointer = ModelCheckpoint('fine_tune_weights.hdf5', verbose=2, monitor='val_loss', save_best_only=True, mode='max')

generator = data_generator()
train_generator = generator.batch_train(train_idx, BATCH_SIZE, SHAPE, augment=True)
validation_generator = generator.batch_train(test_idx, 620, SHAPE, augment=False)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=80,
    verbose=1,
    callbacks=[checkpointer])
show_history(history)
n_list = np.arange(0.1,0.5,0.02)
for idx in test_idx[:3]:
    name0 = idx['name']
    print(idx)
    print(idx['name'])
    print(name0)
%%time
from tqdm import tqdm
TP_data = {}
FP_data = {}
FN_data = {}
F1_best = 0
F1_ther = 0
for threshold in tqdm(n_list):
    F1_sum = 0
    TP_datai = {}
    FP_datai = {}
    FN_datai = {}
    for i in range(28):
        TP_datai[i] = 0
        FP_datai[i] = 0
        FN_datai[i] = 0
    for idx in test_idx[:500]:
        name0 = idx['name']
        generator = data_generator()
        image = generator.load_img(name0, SHAPE)
        score_predict = model.predict(image[np.newaxis,:])
        score_predict = np.array(score_predict)[0]
        label_predict = np.arange(28)[score_predict>=threshold]
        true_label = idx['label']
        true_label = np.array(true_label).astype(int)
        label_predict = set(label_predict)
        true_label = set(true_label)
#         print(label_predict,'label predict')
#         print(true_label,'true_label')
        
        TP = sum(1 for num in label_predict if num in true_label)
#         print(TP,'TP')
        FP = sum(1 for num in label_predict if not num in true_label)
#         print(FP,'FP')
        FN = sum(1 for num in true_label if not num in label_predict)
#         print(FN,'FN')
        TN = 28 - (TP+FP+FN)
        F1_sum += 2*TP/(2*TP+FN+FP)
        
        # count for acc for every label type
#         TP_count = 0
#         FP_count = 0
#         FN_count = 0
        for num in label_predict:
            if num in true_label:
#                 TP_count+=1
                TP_datai[num] += 1
            if num not in true_label:
#                 FP_count+=1
                FP_datai[num] += 1
        for num in true_label:
            if num not in label_predict:
#                 FN_count+=1
                FN_datai[num] += 1
        
        
    if F1_sum>F1_best:
        F1_best = F1_sum
        F1_thre = threshold
        TP_data = TP_datai
        FP_data = FP_datai
        FN_data = FN_datai
        
    print('F1_score_sum: ', F1_sum, 'at threshold: ', threshold)
TP_data = SortedDict(TP_data)
FP_data = SortedDict(FP_data)
FN_data = SortedDict(FN_data)
print('F1_best ', F1_best, '  F1_thre ', F1_thre)
print('TP_data ', TP_data)
print('FP_data ', FP_data)
print('FN_data ', FN_data)
def dict_to_barh(dict_data, title):
    x = list(dict_data.keys())
    y = list(dict_data.values())
    return plt_barh(x, y, title)

dict_to_barh(TP_data, 'TP_data')
dict_to_barh(FP_data, 'FP_data')
dict_to_barh(FN_data, 'FN_data')
submit = pd.read_csv(PATH_BASE+TRAIN_BASE+'sample_submission.csv')
%%time
PATH_TRAIN = PATH_BASE+TRAIN_BASE+'test/'
generator = data_generator()
predicted = []

for name in tqdm(submit['Id']):
    #path = os.path.join('../input/test/', name)
    image = generator.load_img(name, SHAPE)
    score_predict = model.predict(image[np.newaxis,:])[0]
    label_predict = np.arange(28)[score_predict>=F1_thre]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)
submit['Predicted'] = predicted
submit.to_csv('4 channel V2 with rare T2 plus threshold.csv', index=False)