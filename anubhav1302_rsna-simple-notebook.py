import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from matplotlib import pyplot as plt

import pydicom

from tqdm import tqdm

import cv2

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from albumentations import (

    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,

    IAASharpen, IAAEmboss, Flip, OneOf, Compose,VerticalFlip

)
train_df_path='../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv'

test_df_path='../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv'

train_img_path='../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'

test_img_path='../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'

###########

BATCH_SIZE=128

res_h=res_w=128
train_df=pd.read_csv(train_df_path)

test_df=pd.read_csv(test_df_path)

train_df.head()
#Reformat CSV

#Number of Classes=6

class_dict={0:'epidural',1:'intraparenchymal',2:'intraventricular',3:'subarachnoid',4:'subdural',5:'any'}

data_=[] #Format (img_id,class0,...,class6)

for ix in tqdm(range(0,len(train_df),6)):

    tmp=[]

    id_=train_df.loc[ix,'ID'].split('_')

    id_=id_[0]+'_'+id_[1]

    if id_=='ID_6431af929': #Remove corrupt file

        continue

    else:

        tmp.append(id_)

        for i in range(ix,ix+6):

            tmp.append(train_df.loc[i,'Label'])

        data_.append(tmp)

#Lets Check if we got what we wanted

data_[:10]
#Lets check the balance of labels i.e., how many images belongs to some class and and how many belong to none of them.

ids_ones=[]

ids_zeros=[]

for id_ in tqdm(data_):

    if 1 in id_[1:]:

        ids_ones.append([id_[0],id_[1:]])

    else:

        ids_zeros.append(id_[0])
print('Number of Images Belonging to Either of Class: {}'.format(len(ids_ones)))

print('Number of Images belonging to none of the class: {}'.format(len(ids_zeros)))
rows,cols=4,4

fig=plt.figure(figsize=(15,15))

for i in range(1,rows*cols+1):

    tmp=[]

    img_id=data_[100+i]

    id_=img_id[0]

    img=pydicom.read_file(os.path.join(train_img_path,id_+'.dcm')).pixel_array

    fig.add_subplot(rows,cols,i)

    plt.imshow(img,cmap=plt.cm.bone)    

plt.show()
#Now Lets separate ids and labels 

train_labels=np.zeros((len(data_),6))

train_ids=[]

for ix in tqdm(range(len(data_))):

    train_ids.append(data_[ix][0])

    train_labels[ix]=data_[ix][1:]

train_ids,train_labels=shuffle(train_ids,train_labels)



#Split Dataset

t_ids,v_ids,t_labels,v_labels=train_test_split(train_ids,train_labels,test_size=0.2)

print('Size of Train: {}'.format(len(t_ids)))

print('Size of Test: {}'.format(len(v_ids)))

del data_,train_ids,train_labels
def augment_fx(image):

    randx=np.random.randint(0,3)

    if randx==0:

        aug = HorizontalFlip(p=1)

        image = aug(image=image)['image']

        aug = RandomRotate90()

        image = aug(image=image)['image']

        return image

    

    elif randx==1:

        aug = VerticalFlip(p=1)

        image = aug(image=image)['image']

        aug = Transpose() 

        image = aug(image=image)['image']

        return image

    

    elif randx==2:

        aug = VerticalFlip(p=1)

        image = aug(image=image)['image']

        aug = ShiftScaleRotate(p=1)

        image = aug(image=image)['image']

        aug = GridDistortion()

        image = aug(image=image)['image']

        return image

    

    else:

        aug = VerticalFlip(p=1)

        image = aug(image=image)['image']

        aug = HueSaturationValue()

        image = aug(image=image)['image']

        return image
from keras.utils import Sequence



class CustomGenerator(Sequence):

    #Custom Generator for Dataset

    def __init__(self,data,batch_size,res_h,res_w,shuffle=True,image_path=train_img_path,is_train=True):

        self.img_ids=data[0]

        self.label_ids=data[1]

        self.batch_size=batch_size

        self.res_h=res_h

        self.res_w=res_w

        self.shuffle=shuffle

        self.image_path=image_path

        self.is_train=is_train

        self.on_epoch_end()

    

    def __len__(self):

        return int(np.floor(len(self.img_ids)/self.batch_size))

    

    def __getitem__(self,index):

        indexes=self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        tmp_img_ids=[self.img_ids[i] for i in indexes]

        tmp_lbl_ids=[self.label_ids[i] for i in indexes]

        X,y=self.__data_generation(tmp_img_ids,tmp_lbl_ids)

        return X,y

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.img_ids))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def __data_generation(self,image_ids,label_ids):

        image_batch=np.zeros((self.batch_size,self.res_h,self.res_w,1))

        label_batch=np.zeros((self.batch_size,6))

        

        # Read Images

        for ix,img in enumerate(image_ids):

            img=os.path.join(self.image_path,img+'.dcm')

            img=pydicom.read_file(img).pixel_array

            img=img.astype(np.float32)/255.

            img=cv2.resize(img,(self.res_h,self.res_w))

            if self.is_train:

                img=augment_fx(img)

            img=np.expand_dims(img,2)

            image_batch[ix]=img

            

        for ix,lbl in enumerate(label_ids):

            label_batch[ix]=lbl

        

        return image_batch,label_batch
train_generator=CustomGenerator([t_ids,t_labels],BATCH_SIZE,res_h,res_w)

val_generator=CustomGenerator([v_ids,v_labels],BATCH_SIZE,res_h,res_w,is_train=False)

#

train_steps=train_generator.__len__()

val_steps=val_generator.__len__()

print('Train Steps: {}'.format(train_steps))

print('Val Steps: {}'.format(val_steps))
from keras.layers import Conv2D,Dense,Concatenate,Input,GlobalAveragePooling2D,Activation,BatchNormalization

from keras.applications.densenet import DenseNet121

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

import tensorflow as tf

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

model_weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5'
def categorical_focal_loss(gamma=2., alpha=.25):

    """

    Softmax version of focal loss.

           m

      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)

          c=1

      where m = number of classes, c = class and o = observation

    Parameters:

      alpha -- the same as weighing factor in balanced cross entropy

      gamma -- focusing parameter for modulating factor (1-p)

    Default value:

      gamma -- 2.0 as mentioned in the paper

      alpha -- 0.25 as mentioned in the paper

    References:

        Official paper: https://arxiv.org/pdf/1708.02002.pdf

        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:

     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """

    def categorical_focal_loss_fixed(y_true, y_pred):

        """

        :param y_true: A tensor of the same shape as `y_pred`

        :param y_pred: A tensor resulting from a softmax

        :return: Output tensor.

        """



        # Scale predictions so that the class probas of each sample sum to 1

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)



        # Clip the prediction value to prevent NaN's and Inf's

        epsilon = K.epsilon()

        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)



        # Calculate Cross Entropy

        cross_entropy = -y_true * K.log(y_pred)



        # Calculate Focal Loss

        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy



        # Sum the losses in mini_batch

        return K.sum(loss, axis=1)



    return categorical_focal_loss_fixed
mc=ModelCheckpoint('../input/rsna_clf.h5',monitor='val_loss',mode='min',period=1,save_best_only=True)

rlr=ReduceLROnPlateau(monitor='val_loss',min_lr=0.000001,factor=0.2,patience=2)
inp=Input(shape=(res_h,res_w,1))

conv_=Conv2D(3,(3,3),strides=1,padding='same',kernel_initializer='he_normal')(inp)

conv_=Activation('relu')(conv_)

feat_model=DenseNet121(weights=model_weights,include_top=False)(conv_)

gap=GlobalAveragePooling2D()(feat_model)

out=Dense(6,activation='sigmoid')(gap)
model=Model(inp,out)

model.summary()
model.compile(loss='binary_crossentropy',optimizer=Adam(0.001),metrics=['acc'])
history=model.fit_generator(train_generator,steps_per_epoch=train_steps,epochs=4,

                    validation_data=val_generator,validation_steps=val_steps,

                   use_multiprocessing=True,callbacks=[mc,rlr])
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b',color='red', label='Training acc')

plt.plot(epochs, val_acc, 'b',color='blue', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b',color='red', label='Training loss')

plt.plot(epochs, val_loss, 'b',color='blue', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
#Read IDs

test_ids=[]

for ix in tqdm(range(0,len(test_df),6)):

    tmp=[]

    id_=test_df.loc[ix,'ID'].split('_')

    id_=id_[0]+'_'+id_[1]

    test_ids.append(id_)
#Load Model

model.load_weights('../input/rsna_clf.h5')

preds=[]

for id_ in tqdm(test_ids):

    img=os.path.join(test_img_path,id_+'.dcm')

    img=pydicom.read_file(img).pixel_array

    img=img.astype(np.float32)/255.

    img=cv2.resize(img,(res_h,res_w))

    img=np.expand_dims(img,0)

    img=np.expand_dims(img,3)

    preds.append(model.predict(img))
#Submission File

preds=np.reshape(preds,-1)

sub=pd.DataFrame({'ID':test_df['ID'],'Label':preds})

sub.to_csv('submission.csv',index=False)