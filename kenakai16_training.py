import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,sys

import numpy as np

from random import shuffle

import cv2

from tqdm import tqdm



import os

print(os.listdir("../"))
df = pd.read_csv("../input/imet-2019-fgvc6/train.csv")

data=[]

max_lenght=0

id_max_lenght=0

for i in range(len(df)):

    img_path=df.iloc[i]["id"]

    lb=df.iloc[i]["attribute_ids"].split(' ')

    if max_lenght<len(lb):

        max_lenght=len(lb)

        id_max_lenght=img_path

    else:

        max_lenght=max_lenght

    label=[int(i) for i in lb]

    data.append([img_path,label])
data
print("max_lenght:",max_lenght)

print("id_max_lenght:",id_max_lenght)
def creat_square(in_img,value=0):

    img = cv2.imread("../input/imet-2019-fgvc6/train/"+in_img+'.png')

    #grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grey_img=img

    h, w = grey_img.shape[:2]

    edge_square = max(h,w)  

    ground_square=np.ones(shape=[edge_square,edge_square,3])*value

    if h<w:

        x1=np.floor((w-h)/2).astype(int)

        x2=np.floor((w-h)/2+h).astype(int)

        ground_square[x1:x2,:]=grey_img   

    elif h>w:

        x1=np.floor((h-w)/2).astype(int)

        x2=np.floor((h-w)/2+w).astype(int)

        ground_square[:,x1:x2]=grey_img

    else:

        ground_square=grey_img



    return ground_square.astype(np.uint8)
# convert to gray images and resize to 224x224

def resize(img,x_scale=224,y_scale=224):

    img=cv2.resize(img,(x_scale,y_scale))

    #data.astype('uint8')

    return img
shuffle(data)

data
data_trainning=data[:int(len(data)*70/100)]

data_val=data[int(len(data)*70/100):]

#data_test=data[int(len(data)*85/100):]
print("training:",np.array(data_trainning).shape)

print("validation:",np.array(data_val).shape)

#print("test:",np.array(data_test).shape)
from keras.preprocessing.image import ImageDataGenerator
# construct the training image generator for data augmentation

aug = ImageDataGenerator(horizontal_flip=False,

                         rotation_range=90,

                         brightness_range=(0.5,1.3),

                         fill_mode="nearest")
# categorical label

def categorical_label(arr):

    lb=np.zeros(1103)

    for i in arr:

        lb[i]=1

    return lb
def generator(data_txt,batchsize=32,augmentation=False):    

    #map_name=list(zip(list_image,list_label))

    shuffle_data=True

    if shuffle_data:

        shuffle(data_txt)

    #list_image, list_label = zip(*map_name)

    n = len(data_txt)

    i = 0

    while True:

        if i==0:

            shuffle(data_txt)

        X_train=[]

        Y_train=[]       

        for b in range(batchsize):

            img_path=data_txt[i][0]

            #print(img_path)

            img=creat_square(img_path)

            img=resize(img,224,224)

            #img=img.reshape(224,224,1)

            

            label=categorical_label(data_txt[i][1])

            

            X_train.append(img)

            Y_train.append(label)

            

            i = (i+1) % n

        if augmentation is True:

            (images, labels) = next(aug.flow(np.array(X_train),Y_train, batch_size=batchsize))

        else:

            images=X_train

            labels=Y_train

        

        images=np.array(images)

        labels=np.array(labels)

        

        

        images  = images.astype('float32')

        labels  = labels.astype('float32')



        images = images/255

        #print("X_train:",X_train.shape)

        yield (images,labels)
from matplotlib import pyplot

a,b=next(generator(data,10,augmentation=True))

for i in range(0, 9):

    pyplot.subplot(330 + 1 + i)

    pyplot.imshow(a[i], cmap=pyplot.get_cmap('gray'))

pyplot.show()
from matplotlib import pyplot

a,b=next(generator(data,10,augmentation=False))

for i in range(0, 9):

    pyplot.subplot(330 + 1 + i)

    pyplot.imshow(a[i], cmap=pyplot.get_cmap('gray'))

pyplot.show()
import keras

from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D,add,GlobalAveragePooling2D

from keras import regularizers, optimizers

from keras import applications

from keras.models import Model,load_model

from keras.optimizers import RMSprop,Adam,SGD

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping

from keras.callbacks import  TensorBoard

from keras import backend as K

import tensorflow as tf
def creat_model():

    base_model=applications.ResNet50(weights=None,include_top=False,input_shape=(224, 224, 3))

    base_model.load_weights("../input/pretrain/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

    

    out = base_model.output

    out = Dropout(0.5)(out)

    out = BatchNormalization()(out)

    out = GlobalAveragePooling2D()(out)

    out = BatchNormalization()(out)

    out = Dropout(0.5)(out)

    out = BatchNormalization()(out)

    predictions = Dense(1103, activation= 'sigmoid')(out)

    custom_model = Model(inputs = base_model.input, outputs = predictions)

    return custom_model
model=creat_model()
model.summary()
for layer in model.layers[:-7]:

    layer.trainable = False
for layer in model.layers[-7:]:

    layer.trainable = True
model.summary()
## precision, recall, and f measure

#def precision(y_true, y_pred):

#    # Calculates the precision

#    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

#    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

#    precision = true_positives / (predicted_positives + K.epsilon())

#    return precision

#

#def recall(y_true, y_pred):

#    # Calculates the recall

#    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

#    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

#    recall = true_positives / (possible_positives + K.epsilon())

#    return recall

#

#def fbeta_score(y_true, y_pred, beta=2):

#    # Calculates the F score, the weighted harmonic mean of precision and recall.

#    if beta < 0:

#        raise ValueError('The lowest choosable beta is zero (only precision).')

#    

#    # If there are no true positives, fix the F score at 0 like sklearn.

#    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

#        return 0

#

#    p = precision(y_true, y_pred)

#    r = recall(y_true, y_pred)

#    bb = beta ** 2

#    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

#    return fbeta_score

#

#def fmeasure(y_true, y_pred):

#    # Calculates the f-measure, the harmonic mean of precision and recall.

#    return fbeta_score(y_true, y_pred, beta=2)
beta_f2=2



# if gamma == 0.0:

#     F2_THRESHOLD = 0.1

# elif gamma == 1.0:

#     F2_THRESHOLD = 0.2

# else:

#     F2_THRESHOLD = 0.3



# print(F2_THRESHOLD)

    

def f2(y_true, y_pred):

    #y_pred = K.round(y_pred)

#     y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), F2_THRESHOLD), K.floatx())

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=1)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=1)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=1)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=1)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f2 = (1+beta_f2**2)*p*r / (p*beta_f2**2 + r + K.epsilon())

    f2 = tf.where(tf.is_nan(f2), tf.zeros_like(f2), f2)

    return K.mean(f2)
#from keras import backend as K

#import tensorflow as tf

#

#import dill

#

#

#def binary_focal_loss(gamma=2., alpha=0.25):

#

#    def binary_focal_loss_fixed(y_true, y_pred):

#        

#        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

#        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

#

#        epsilon = K.epsilon()

#        # clip to prevent NaN's and Inf's

#        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)

#        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

#

#        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \

#               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

#

#    return binary_focal_loss_fixed
# focal loss

#def focal_loss(gamma=2., alpha=4.):#



#    gamma = float(gamma)

#    alpha = float(alpha)

#

#    def focal_loss_fixed(y_true, y_pred):

#        epsilon = 1.e-7

#        y_true = tf.convert_to_tensor(y_true, tf.float32)

#        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

#

#        model_out = tf.add(y_pred, epsilon)

#        ce = tf.multiply(y_true, -tf.log(model_out))

#        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))

#        fl = tf.multiply(alpha, tf.multiply(weight, ce))

#        reduced_fl = tf.reduce_max(fl, axis=1)

#        return tf.reduce_mean(reduced_fl)

#    return focal_loss_fixed



def focal_loss(gamma=2., alpha=2):

    gamma = 2.0

    epsilon = K.epsilon()

    def focal_loss(y_true, y_pred):

        pt = y_pred * y_true + (1-y_pred) * (1-y_true)

        pt = K.clip(pt, epsilon, 1-epsilon)

        CE = -K.log(pt)

        FL = K.pow(1-pt, gamma) * CE

        loss = K.sum(FL, axis=1)

        return loss

    return focal_loss
import tensorflow as tf

import keras.backend as K



def f1_macro(y_true, y_pred):

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)
os.mkdir("../checkpoint/")
os.listdir("../")
model.compile(optimizer=Adam(lr=1e-3),

                     loss="binary_crossentropy",

                     metrics=['accuracy',f1_macro,f2])

#custom_model.compile(optimizer=Adam(lr=1e-3), loss=focal_loss(), metrics=['accuracy',f1_macro,f2])
tensorboard = TensorBoard(log_dir ='../training',

                          write_graph=True,

                          write_images=True,

                          update_freq='batch'

                            )

reduce_lr = ReduceLROnPlateau(monitor='val_acc', 

                              factor=0.5,patience=5, min_lr=0.0001)

early_stopping = EarlyStopping(monitor='acc', min_delta=0, patience=10, verbose=1)

checkpoint = ModelCheckpoint('../checkpoint/weights.{epoch:02d}-{val_acc:.5f}_binary_crossentropy_f1_macro.hdf5',

                             monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
batch_size_train=32

batch_size_val=32

history=model.fit_generator(generator(data_trainning,batch_size_train),

                steps_per_epoch=max(1, len(data_trainning)//batch_size_train),

                validation_data=generator(data_val,batch_size_val),#

                validation_steps=max(1, len(data_val)//batch_size_val),#

                epochs=1,

                callbacks=[reduce_lr,checkpoint,tensorboard,reduce_lr])
os.mkdir("../model")
model.save("../model/model.h5")
os.listdir("../model/")
model_s2=creat_model()
model_s2.load_weights("../model/model.h5")
for layer in model_s2.layers[:]:

    layer.trainable = True
model_s2.summary()
model_s2.compile(optimizer=Adam(lr=1e-3),

                     loss="binary_crossentropy",

                     metrics=['accuracy',f1_macro,f2])

#custom_model.compile(optimizer=Adam(lr=1e-3), loss=focal_loss(), metrics=['accuracy',f1_macro,f2])
batch_size_train=32

batch_size_val=32

history=model_s2.fit_generator(generator(data_trainning,batch_size_train),

                steps_per_epoch=max(1, len(data_trainning)//batch_size_train),

                validation_data=generator(data_val,batch_size_val),#

                validation_steps=max(1, len(data_val)//batch_size_val),#

                epochs=10,

                callbacks=[reduce_lr,checkpoint,tensorboard,reduce_lr])
model_s2.save("../model_0.3478.h5")
#print(K.eval(model_s2.optimizer.lr))

#lr=(K.eval(model_s2.optimizer.lr

#model_s2.compile(optimizer=Adam(lr=lr),

#                     loss=focal_loss(),

#                     metrics=['accuracy',f1_macro,f2])

#tensorboard = TensorBoard(log_dir ='training',

#                          write_graph=True,

#                          write_images=True,

#                          update_freq='batch'

#                            )

#

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', 

#                              factor=0.5,patience=5, min_lr=0.0001)

#

#early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

#checkpoint = ModelCheckpoint('../weights.{epoch:02d}-{val_acc:.5f}_focal_f1_macro.hdf5',

#                             monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

#batch_size_train=32

#batch_size_val=32

#history=model_s2.fit_generator(generator(data_trainning,batch_size_train),

#                steps_per_epoch=max(1, len(data_trainning)//batch_size_train),

#                validation_data=generator(data_val,batch_size_val),#

#                validation_steps=max(1, len(data_val)//batch_size_val),#

#                epochs=10,

#                callbacks=[reduce_lr,checkpoint,tensorboard,reduce_lr])
def my_f2(y_true, y_pred):

    assert y_true.shape[0] == y_pred.shape[0]



    tp = np.sum((y_true == 1) & (y_pred == 1),axis=1)

    tn = np.sum((y_true == 0) & (y_pred == 0),axis=1)

    fp = np.sum((y_true == 0) & (y_pred == 1),axis=1)

    fn = np.sum((y_true == 1) & (y_pred == 0),axis=1)

    

    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f2 = (1+beta_f2**2)*p*r / (p*beta_f2**2 + r + 1e-15)



    return np.mean(f2)



def find_best_fixed_threshold(preds, targs, do_plot=True):

    score = []

    thrs = np.arange(0, 0.5, 0.01)

    for thr in tqdm(thrs):

        score.append(my_f2(targs, (preds > thr).astype(int) ))

    score = np.array(score)

    pm = score.argmax()

    best_thr, best_score = thrs[pm], score[pm].item()

    print(f'thr={best_thr:.3f}', f'F2={best_score:.3f}')

    if do_plot:

        plt.plot(thrs, score)

        plt.vlines(x=best_thr, ymin=score.min(), ymax=score.max())

        plt.text(best_thr+0.03, best_score-0.01, f'$F_{2}=${best_score:.3f}', fontsize=14);

        plt.show()

    return best_thr, best_score
def creat_square_test(in_img,value=0):

    img = cv2.imread("../input/imet-2019-fgvc6/test/"+in_img+'.png')

    #grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grey_img=img

    h, w = grey_img.shape[:2]

    edge_square = max(h,w)  

    ground_square=np.ones(shape=[edge_square,edge_square,3])*value

    if h<w:

        x1=np.floor((w-h)/2).astype(int)

        x2=np.floor((w-h)/2+h).astype(int)

        ground_square[x1:x2,:]=grey_img   

    elif h>w:

        x1=np.floor((h-w)/2).astype(int)

        x2=np.floor((h-w)/2+w).astype(int)

        ground_square[:,x1:x2]=grey_img

    else:

        ground_square=grey_img



    return ground_square.astype(np.uint8)
model_s2.load_weights("../model_0.3478.h5")
batch=512

n_val = round(len(data_val))//batch

fullValGen =generator(data_val,batch)

lastFullValPred = np.empty((0, 1103))

lastFullValLabels = np.empty((0, 1103))

for i in tqdm(range(n_val+1)): 

    #print(i)

    im, lbl = next(fullValGen)

    scores = model_s2.predict(im)

    lastFullValPred = np.append(lastFullValPred, scores, axis=0)

    lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)

print(lastFullValPred.shape, lastFullValLabels.shape)

best_thr, best_score = find_best_fixed_threshold(lastFullValPred, lastFullValLabels, do_plot=False)

print("best_thr={}, best_score={}".format(best_thr, best_score))
from tqdm import tqdm

submit = pd.read_csv('../input/imet-2019-fgvc6/sample_submission.csv')

result_predict=[]

for i, name in tqdm(enumerate(submit['id'])):

    image = creat_square_test(name)

    image = resize(image)

    score_predict = model_s2.predict(image[np.newaxis]/255)

    # print(score_predict)

    label_predict = np.arange(1103)[score_predict[0]>=best_thr]

    print("label_predict:",label_predict)

    str_predict_label = ' '.join(str(l) for l in label_predict)

    result_predict.append(str_predict_label)
result_predict
submit = pd.read_csv('../input/imet-2019-fgvc6/sample_submission.csv')

submit['attribute_ids'] = result_predict

submit.to_csv('submission.csv', index=False)
submit