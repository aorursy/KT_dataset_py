import numpy as np
import pandas as pd
import glob
import cv2 
from PIL import Image
from tqdm import tqdm
images=[]
display=[]
labels=[]
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/0/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("0")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/1/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("1")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/2/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("2")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/3/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("3")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/4/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("4")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/5/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("5")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/6/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("6")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/7/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("7")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/8/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("8")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/9/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("9")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/a/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("a")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/b/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("b")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/c/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("c")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/d/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("d")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/e/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("e")
    i=i+1
    if i==1280:
        break
i=0
for img in tqdm(glob.glob("../input/landmark-retrieval-2020/train/f/*/*/*.jpg")):
    image=cv2.imread(img)
    ar=Image.fromarray(image,'RGB')
    res=ar.resize((64,64))
    images.append(np.array(res))
    labels.append("f")
    i=i+1
    if i==1280:
        break
len(images)
images=np.array(images)
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
labels=lab.fit_transform(labels)
labels=np.array(labels)
np.save("image",images)
np.save("labels",labels)
len(image)
image=np.load("image.npy",allow_pickle=True)
labels=np.load("labels.npy",allow_pickle=True)
import matplotlib.pyplot as plt
figure=plt.figure(figsize=(25,25))
ax=figure.add_subplot(231)
ax.imshow(image[1])
bx=figure.add_subplot(232)
bx.imshow(image[60])
cx=figure.add_subplot(233)
cx.imshow(image[1000])
dx=figure.add_subplot(234)
dx.imshow(image[6000])
ex=figure.add_subplot(235)
ex.imshow(image[10000])
fx=figure.add_subplot(236)
fx.imshow(image[19000])
plt.show()
s=np.arange(image.shape[0])
np.random.shuffle(s)
image=image[s]
labels=labels[s]
num_classes=len(np.unique(labels))
len_data=len(image)
x_train,x_test=image[(int)(0.1*len_data):],image[:(int)(0.1*len_data)]
y_train,y_test=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
import keras
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.layers import Activation, Convolution2D, Dropout, Conv2D,AveragePooling2D, BatchNormalization,Flatten,GlobalAveragePooling2D
from keras import layers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.applications.vgg19 import VGG19
base_model = VGG19(weights='imagenet',include_top=False, input_shape=(64,64,3))
x = base_model.output
x = Flatten()(x)
x=Dense(500, activation='relu')(x)
x=Dropout(0.2)(x)
predictions = Dense(16, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(x_train,y_train,batch_size=128,epochs=5,verbose=1,validation_split=0.1)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
figure=plt.figure(figsize=(15,15))
ax=figure.add_subplot(121)
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.legend(['Training Accuracy','Val Accuracy'])
bx=figure.add_subplot(122)
bx.plot(history.history['loss'])
bx.plot(history.history['val_loss'])
bx.legend(['Training Loss','Val Loss'])