import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import os

import glob

from keras.layers import Conv2D,Dense,Dropout,Flatten,Input

from keras.models import Model

from keras.layers.pooling import MaxPooling2D,GlobalMaxPooling2D,AveragePooling2D

import seaborn as sns

import time
Train_df = pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Train.csv')

Test = pd.read_csv("/kaggle/input/gtsrb-german-traffic-sign/Test.csv")
Train_df.info
Train_df.empty
train_img_dim = Train_df[['Width','Height']]
g = sns.JointGrid(x="Width", y="Height", data=train_img_dim)

sns.kdeplot(train_img_dim.Width, train_img_dim.Height, cmap="Reds",

        shade=False, shade_lowest=False, ax=g.ax_joint)

sns.distplot(train_img_dim.Width, kde=True, hist=False, color="r", ax=g.ax_marg_x, label='Train distribution')

sns.distplot(train_img_dim.Height, kde=True, hist=False, color="r", ax=g.ax_marg_y, vertical=True)

g.fig.set_figwidth(30)

g.fig.set_figheight(12)

plt.show();
Train = Train_df.iloc[:,:].values

Classid = Train[:,6]

Train_path = Train[:,7]

Test = Test.iloc[:,:].values

Test_path = Test[:,7]

Test_class = Test[:,6]
def image_loader(Path):

    data = []

    home = "/kaggle/input/gtsrb-german-traffic-sign/"

    for p in Path:

        img = cv2.imread(home+p)

        img = cv2.resize(img,(32,32))

        img = img/255.0

        data.append(img)

    return np.array(data)
Train_data = image_loader(Train_path)

plt.imshow(Train_data[0])

print(Train_data.shape)
Test_data = image_loader(Test_path)
nr_channel = Train_data[3].shape[2]
def one_hot_encoder(new_labels,labels):

    for i in range(new_labels.shape[0]):  #one_hot_encoder

        num = int(labels[i])

        new_labels[i,num] = 1

    return new_labels

def one_hot_decode(t_d):

    t_decoded=np.zeros([t_d.shape[0],1],int)   #one_hot_decoder

    for i in range(t_d.shape[0]):

        for j in range(t_d.shape[1]):

            if t_d[i,j]==1:

                t_decoded[i,0]=j+1

    return t_decoded
num_classes = 43

train_length = len(Classid)
Class_id_encoded = np.zeros([train_length,num_classes],int)

Class_id_encoded = np.array(one_hot_encoder(Class_id_encoded,Classid))

print(Class_id_encoded[0].shape)

Test_class_enc = np.zeros([Test_data.shape[0],num_classes],int)

Test_class_enc = np.array(one_hot_encoder(Test_class_enc,Test_class))
input_layer = Input(shape=(None,None,3))

m = Conv2D(32,(5,5),strides=(1,1),activation='relu')(input_layer)

m = Dropout(0.4)(m)

m = MaxPooling2D(strides=(2,2))(m)

m = Conv2D(64,(5,5),strides=(1,1),activation='relu')(m)

m = Dropout(0.3)(m)

m = MaxPooling2D(strides=(2,2))(m)

m = Conv2D(128,(5,5),strides=(1,1),activation='relu')(m)

m = Dropout(0.2)(m)

m = GlobalMaxPooling2D()(m)

m = Dense(84,activation='relu')(m)

output_layer = Dense(43,activation='softmax')(m)

LeNet = Model(inputs=input_layer,outputs=output_layer)

LeNet.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

LeNet.summary()
Train_attributes = LeNet.fit(Train_data,Class_id_encoded,batch_size=256,epochs=30)
test_attributes = LeNet.evaluate(x=Test_data, y=Test_class_enc, batch_size=256)

print("Test_data_loss:",test_attributes[0],"Test_data_accuracy:",test_attributes[1])
plt.plot(Train_attributes.history['loss'],'b')

plt.ylabel('Training_Loss')

plt.xlabel('Iterations')

plt.show()
plt.plot(Train_attributes.history['accuracy'],'b')

plt.ylabel('Training_accuracy')

plt.xlabel('Iterations')

plt.show()