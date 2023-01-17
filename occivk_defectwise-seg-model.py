import cv2,os

import numpy as np

from tqdm.notebook import tqdm as tqdm
!pip install segmentation_models
import segmentation_models as sm
model = sm.FPN('efficientnetb1',input_shape=(256,256,3),classes=2,activation='softmax')

model.summary()
os.listdir('../input/defectwise-dataset-onions/defectwise_dataset')
def getData(im, classes, w,h):

    img = cv2.imread(os.path.join(path+'images',im))

    img = np.float32(cv2.resize(img, (w,h)))

    mask = cv2.imread(os.path.join(path+'masks',im),0)

    mask = cv2.resize(mask, ( w , h ))

    ret,mask = cv2.threshold(mask,127,1,cv2.THRESH_BINARY)

    seg_labels = np.zeros((h,w,classes))

    for i in range(h):

        for j in range(w):

            if mask[i,j]>0:

                seg_labels[i,j,1]=1

            else:

                seg_labels[i,j,0]=1

    return img, seg_labels





path = '../input/defectwise-dataset-onions/defectwise_dataset/train/'

classes = 2

w,h = 256,256

X_train = []

Y_train = []

for im in tqdm(os.listdir(path+'images')):

    if im.endswith('.jpg'):

        img, mask = getData(im, classes, w,h)

        X_train.append(img)

        Y_train.append(mask)

X_train, Y_train = np.array(X_train),np.array(Y_train)

print(X_train.shape, Y_train.shape)
path = '../input/defectwise-dataset-onions/defectwise_dataset/val/'

classes = 2

w,h = 256,256

X_val = []

Y_val = []

for im in tqdm(os.listdir(path+'images')):

    if im.endswith('.jpg'):

        img, mask = getData(im, classes, w,h)

        X_val.append(img)

        Y_val.append(mask)

X_val, Y_val = np.array(X_val),np.array(Y_val)

print(X_val.shape, Y_val.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
from keras.losses import binary_crossentropy

from keras.optimizers import SGD,Adam,RMSprop

from keras import backend as K

from datetime import datetime



opt = RMSprop(lr=0.01)

total_loss = sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5),sm.metrics.FScore(threshold=0.5)]

model.compile(loss=total_loss,optimizer=opt, metrics=metrics)
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

from keras.models import load_model

from datetime import datetime

model_name = "defecfwise_Segmentation_Model_"+str(256_2)+".h5"



callback = [ReduceLROnPlateau(patience=5, verbose=1),

            ModelCheckpoint(model_name,

                            save_best_only=True,

                            save_weights_only=False),ModelCheckpoint("defectwise_Segmentation_Model_"+str(256_2)+"_weights.h5",

                            save_best_only=True,

                            save_weights_only=True)]



history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),batch_size=4,verbose=1,

                   callbacks=callback,epochs=100)
os.listdir('../input/defectwise-dataset-onions/defectwise_dataset/test')
def color_splash(image, mask):

    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    # We're treating all instances as one, so collapse the mask into one layer

    # mask = (np.sum(mask, -1, keepdims=True) >= 1)

    # Copy color pixels from the original color image where mask is set

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if mask.shape[0] > 0:

        splash = np.where(mask, image, gray).astype(np.uint8)

        msk = mask.copy()

        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)

        ret,thresh = cv2.threshold(msk,127,255,cv2.THRESH_BINARY)

        contours = cv2.findContours(thresh, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[0]

        cv2.drawContours(splash, contours, -1, (0,255,0),3)

    else:

        splash = gray

    return splash
import glob

import skimage

import matplotlib.pyplot as plt 

img_path = glob.glob('../input/defectwise-dataset-onions/defectwise_dataset/test/images/*.jpg')



for image in img_path:

    img = cv2.imread(image)

    im = cv2.resize(img,(256,256))

    name = image.split('/')[-1]

    I = im.reshape([1,256,256,3])

    preds = model.predict(I)

    preds = np.argmax(preds,axis=3)



    prediction_mask = preds.reshape([256,256])

    prediction_mask = np.uint8(prediction_mask*255)

    prediction_mask = cv2.resize(prediction_mask,(img.shape[1],img.shape[0]))

    print(img.shape[0],img.shape[1],prediction_mask.shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    seg = cv2.bitwise_and(img, img, mask=prediction_mask)

    splash = color_splash(img, prediction_mask)

    plt.figure(figsize=(9,9))

    plt.subplot(1,2,1)

    plt.imshow(splash)

    plt.subplot(1,2,2)

    plt.imshow(img)

    plt.show()