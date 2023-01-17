import glob
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization,Add,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, concatenate
def UNet():
    input_size = (288, 480, 5)
    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    coarse_decoder_model = Model(inputs=inp, outputs=[conv10])
    #fine_input_merge = concatenate([inp,conv9], axis=3)
    #fine_conv9 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(fine_input_merge)
    #fine_conv9 = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(fine_conv9)
    #fine_docoder_model = Model(inputs=inp,outputs=fine_conv9)
    return coarse_decoder_model
trainPointNames=[]
for dirname, _, filenames in os.walk('../input/davis-pointannotation-dataset/Annotations/Point'):
    for filename in filenames:
           trainPointNames.append(os.path.join(dirname, filename))
#Train
trainMask = []
trainImg = []
trainPoint = []
trainNPoint = []
trainPointNames.sort()
for j in range(len(trainPointNames)):
    trainPoint.append(trainPointNames[j])
    Imgpath = trainPointNames[j].replace('/Annotations', '/JPEGImages')
    Imgpath = Imgpath.replace('/Point', '/480p')
    Imgpath = Imgpath.replace('.png','.jpg')
    trainImg.append(Imgpath)
    maskpath = trainPointNames[j].replace('/Point','/480p')
    trainMask.append(maskpath)
#valid
validMask = []
validImg = []
validPoint = []
validNPoint = []

for j in range(len(trainPointNames)):
    validPoint.append(trainPointNames[j])
    Imgpath = trainPointNames[j].replace('/Annotations', '/JPEGImages')
    Imgpath = Imgpath.replace('/Point', '/480p')
    Imgpath = Imgpath.replace('.png', '.jpg')
    validImg.append(Imgpath)
    maskpath = trainPointNames[j].replace('/Point', '/480p')
    validMask.append(maskpath)
    
train_x = np.zeros((len(trainImg),288,480, 5))
valid_x = np.zeros((len(validImg),288,480, 5))
train_mask_y = np.zeros((len(trainMask),288,480))
valid_mask_y = np.zeros((len(validMask),288,480))
def inverse_1_to_0(img):
    res = np.zeros(img.shape)
    for i,width in enumerate(img):
        for j,height in enumerate(width):
            if(height==0):
                res[i,j]=1
            else:
                res[i,j]=0
    return np.array(res)
#train

for index in range(len(trainImg)):
    mask = cv2.imread(trainMask[index], cv2.IMREAD_GRAYSCALE)
    dstmask = cv2.resize(mask, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    img = cv2.imread(trainImg[index], cv2.IMREAD_COLOR)
    dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    Point = cv2.imread(trainPoint[index], cv2.IMREAD_GRAYSCALE)
    dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    tmask = np.array(dstmask)
    timg = np.array(dstimg)
    train_Point = np.array(dstPoint)
    train_NPoint = inverse_1_to_0(dstPoint)
    train_x[index,:,:,0:3] = timg
    train_x[index,:,:,3] = train_Point
    train_x[index,:,:,4] = train_NPoint
    train_mask_y[index,:,:] = tmask
dstPoint.shape
#valid
for index in range(len(validImg)):
    mask = cv2.imread(validMask[index], cv2.IMREAD_GRAYSCALE)
    dstmask = cv2.resize(mask, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    img = cv2.imread(validImg[index], cv2.IMREAD_COLOR)
    dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    Point = cv2.imread(validPoint[index], cv2.IMREAD_GRAYSCALE)
    dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    vmask = np.array(dstmask)
    vimg = np.array(dstimg)
    valid_Point =np.array(dstPoint)
    valid_x[index, :, :, 0:3] = vimg
    valid_x[index, :, :, 3] = valid_Point
    valid_NPoint = inverse_1_to_0(dstPoint)
    valid_x[index, :, :, 4] = valid_NPoint
    valid_mask_y[index,:,:] = vmask

train_x = train_x.reshape(len(train_x), 288, 480, 4).astype('float32')/255.0
valid_x = valid_x.reshape(len(valid_x), 288, 480, 4).astype('float32')/255.0
train_mask_y = train_mask_y.reshape(len(train_mask_y), 288, 480, 1).astype('float32')/255.0
valid_mask_y = valid_mask_y.reshape(len(valid_mask_y), 288, 480, 1).astype('float32')/255.0
model = UNet()
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
train_x.shape
model =UNet()
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x=train_x, y=train_mask_y, epochs=1, batch_size=1,verbose=1)
teacher_train_pred = np.zeros((len(trainImg),288,480, 5))
teacher_valid_pred = np.zeros((len(validImg),288,480, 5))
teacherPoint = model.predict(train_x)
for index in range(len(trainImg)):
    teacher_train_pred[index,:,:,0:4] = train_x[index]
    teacher_train_pred[index,:,:,4] = teacherPoint[index,:,:,0]

# In[ ]:
teacherPoint = model.predict(valid_x)
for index in range(len(validImg)):
    teacher_valid_pred[index,:,:,0:4] = valid_x[index]
    teacher_valid_pred[index,:,:,4] = teacherPoint[index,:,:,0]

smodel = fine_decoder()
smodel.summary()
fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(results.history['loss'], 'r')
ax[0, 1].set_title('acc')
ax[0, 1].plot(results.history['acc'], 'b')

ax[1, 0].set_title('val_loss')
ax[1, 0].plot(results.history['val_loss'], 'r--')
ax[1, 1].set_title('val_acc')
ax[1, 1].plot(results.history['val_acc'], 'b--')

# summarize history for accuracy
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

