import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import cv2
from keras.models import Model,Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.optimizers import SGD,Adam
IMG_ROW=IMG_COL=64
IMG_CHANNEL=3
IMG_DIR='../input/sample/images/'
labeldata=pd.read_csv('../input/sample_labels.csv')
print(labeldata.columns)
labelset=set()
for labelstr in labeldata['Finding Labels']:
    labelarr=labelstr.split('|')
    for label in labelarr:
        labelset.add(label)
labellist=list(labelset)
print(len(labellist))
labelsequence=np.zeros((len(labeldata),len(labellist)))
for i,labelstr in enumerate(labeldata['Finding Labels']):
    labelarr=labelstr.split('|')
    for label in labelarr:
        index1=labellist.index(label)
        labelsequence[i][index1]=1
def read_img(imgpath):
    img=cv2.imread(imgpath)
    img=cv2.resize(img,(IMG_ROW,IMG_COL))
    return img
trainimg=[]
labelarr1=[]
for i,(path,labelstr) in enumerate(zip(labeldata['Image Index'],labeldata['Finding Labels'])):
    imgpath=IMG_DIR+path
    img=read_img(imgpath)
    labelarr=labelstr.split('|')
    for label in labelarr:
        labelindex=labellist.index(label)
        trainimg.append(img)
        labelarr1.append(labelindex)
trainimg=np.asarray(trainimg)
labelarr1=np.asarray(labelarr1)
print(trainimg.shape)
print(labelarr1.shape)
def build_generator(latent_dim=100):
    noise=Input(shape=(latent_dim,))

    x=Dense(128*int(IMG_ROW/4)*int(IMG_COL/4),activation='relu')(noise)
    x=Reshape((int(IMG_ROW/4),int(IMG_COL/4),128))(x)
    x=BatchNormalization(momentum=0.8)(x)
    x=UpSampling2D((2,2))(x)
    x=Conv2D(128,kernel_size=(3,3),padding='same',activation='relu')(x)
    x=BatchNormalization(momentum=0.8)(x)
    x=UpSampling2D((2,2))(x)
    x=Conv2D(64,kernel_size=(3,3),padding='same',activation='relu')(x)
    x=BatchNormalization(momentum=0.8)(x)
    x=Conv2D(3,kernel_size=(3,3),padding='same',activation='relu')(x)
    model=Model(noise,x)
    label=Input(shape=(1,),dtype='int32')
    label_embedding=Flatten()(Embedding(10,100)(label))
    model_input=Multiply()([noise,label_embedding])
    img=model(model_input)
    return Model([noise,label],img)
def build_discriminator():
    img=Input(shape=(IMG_ROW,IMG_COL,IMG_CHANNEL))
    x=Conv2D(16, kernel_size=3, strides=2, padding="same")(img)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.25)(x)
    x=Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
    x=ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.25)(x)
    x=BatchNormalization(momentum=0.8)(x)
    x=Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.25)(x)
    x=BatchNormalization(momentum=0.8)(x)
    x=Conv2D(128, kernel_size=3, strides=1, padding="same")(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.25)(x)
    x=Flatten()(x)
    model=Model(img,x)
    features=model(img)
    validity = Dense(1, activation="sigmoid")(features)
    label=Dense(len(labellist)+1,activation='softmax')(features)
    return Model(img,[validity,label])
def combined_generator_discriminator():
    optimizer = Adam(0.0002, 0.5)
    losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
    discriminator = build_discriminator()
    discriminator.compile(loss=losses,
                               optimizer=optimizer,
                               metrics=['accuracy'])
    generator = build_generator()
    noise = Input(shape=(100,))
    label = Input(shape=(1,))
    img = generator([noise, label])
    discriminator.trainable = False
    valid, target_label =discriminator(img)
    combined = Model([noise, label], [valid, target_label])
    combined.compile(loss=losses,
                          optimizer=optimizer)
    return generator,discriminator,combined
def train(batch_size=200,epochs=10000):
    generator,discriminator,combined=combined_generator_discriminator()
    valid=np.ones((batch_size,1))
    fake=np.zeros((batch_size,1))
    g_loss_epochs=np.zeros((epochs,1))
    d_loss_epochs=np.zeros((epochs,1))
    for epoch in range(epochs):
        idx=np.random.randint(0,trainimg.shape[0],batch_size)
        imgs=trainimg[idx]
        noise=np.random.normal(0,1,(batch_size,100))
        sampled_labels=np.random.randint(0,len(labellist),(batch_size,1))
        genimgs=generator.predict([noise,sampled_labels])
        img_labels=labelarr1[idx]
        fake_labels=len(labellist)*np.ones(batch_size)
        d_loss_real=discriminator.train_on_batch(imgs,[valid,img_labels])
        d_loss_fake=discriminator.train_on_batch(genimgs,[fake,fake_labels])
        d_loss=0.5*np.add(d_loss_real,d_loss_fake)
        g_loss=combined.train_on_batch([noise,sampled_labels],[valid,sampled_labels])
        g_loss_epochs[epoch] = g_loss[0]
        d_loss_epochs[epoch] = d_loss[0]
        if(epoch%50==0):
            print ("Epoch: %d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
        if epoch % 100 == 0:
            samples = len(labellist)
            z = np.random.normal(loc=0, scale=1, size=(samples, 100))
            labels = np.arange(0, len(labellist)).reshape(-1, 1)
            x_fake = generator.predict([z, labels])
            for k in range(samples):
                plt.subplot(4, int(len(labellist)/4)+1, k + 1, xticks=[], yticks=[])
                plt.imshow(x_fake[k]/255)
                plt.title(labellist[k])
            plt.show()
        
    return g_loss_epochs,d_loss_epochs,generator
g_loss_epochs,d_loss_epochs,generator=train()
generator.save_weights('model.h5')
plt.plot(g_loss_epochs)
plt.plot(d_loss_epochs)