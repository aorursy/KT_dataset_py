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
from imgaug import augmenters as iaa
%matplotlib inline
os.listdir('../input')
PATH_BASE = '../input/'
TRAIN_BASE = 'human-protein-atlas-image-classification/'
MODEL_BASE = 'inceptionresnetv2-pre-train-model/'
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

from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(labels, test_size=0.2)
print('train: ' + str(len(train_idx)) + '\n'+ 'validation: ' + str(len(test_idx)))
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

            #batch_images = 提取images存成array, shape=(batch_size, shpae[0], shape[1], shpae[2]) = batch_images(能夠改名字嗎?例如trainX之類的)
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
        BY = (B+Y)
        BY[BY>255] = 255
        image = np.stack((R, G, BY) ,axis=-1)
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
#test block
generator_test = data_generator()
#np.random.seed(43) #just for test purpose
K0 = generator_test.batch_train(train_idx, 1,(500,500,3),True)
a,b = next(K0)
A = np.squeeze(a)
plt.figure(figsize=(20,10))
plt.imshow(A)
b
from keras import applications
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
import tensorflow as tf
import keras.backend as K
K.clear_session()
SHAPE = (299,299,3)
BATCH_SIZE = 10

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
#Use this cell to read model & weight
model = load_model(PATH_BASE+MODEL_BASE+'fine_tune_weights.hdf5', custom_objects={'f1':f1})
model.summary()
for layer in model.layers[:780]:
    layer.trainable =False
for layer in model.layers[:]:
    layer.trainable =True
model.layers[167].trainable
model.compile(
    loss=[binary_focal_loss],  
    optimizer=Adam(1e-4),
    metrics=['acc', f1])
checkpointer = ModelCheckpoint('fine_tune_weights.hdf5', verbose=2, monitor='val_acc', save_best_only=True)

generator = data_generator()
train_generator = generator.batch_train(train_idx, BATCH_SIZE, SHAPE, augment=True)
validation_generator = generator.batch_train(test_idx, 256, SHAPE, augment=False)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=120, 
    verbose=1,
    callbacks=[checkpointer])
submit = pd.read_csv(PATH_BASE+TRAIN_BASE+'sample_submission.csv')
from tqdm import tqdm
%%time
PATH_TRAIN = PATH_BASE+TRAIN_BASE+'test/'
generator = data_generator()
predicted = []

for name in tqdm(submit['Id']):
    #path = os.path.join('../input/test/', name)
    image = generator.load_img(name, SHAPE)
    score_predict = model.predict(image[np.newaxis,:])[0]
    label_predict = np.arange(28)[score_predict>=0.2]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)
submit['Predicted'] = predicted
submit.to_csv('focal_loss_with_pre_train_V2_submission.csv', index=False)