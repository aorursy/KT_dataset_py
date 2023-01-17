from PIL import Image, ImageDraw, ImageFont

from os import listdir

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

from skimage.feature import hog

import os

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from keras import backend as K

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from tensorflow.python import keras

from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization,Input

from keras.models import Model,load_model

from IPython.display import SVG

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

#%matplotlib inline

print(os.listdir("../input/"))

InputPath = "../input/artificial-lunar-rocky-landscape-dataset/images/"

# Any results you write to the current direct
fontsize = 50



# From https://www.google.com/get/noto/

!wget -q --show-progress https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip

!unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf > NotoSansCJKjp-Regular.otf

!rm NotoSansCJKjp-hinted.zip



font = ImageFont.truetype('./NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')
df_train = pd.read_csv('../input/train.csv')

unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../input/unicode_translation.csv').values}

unicode_map
df_train.isnull().sum()
# This function takes in a filename of an image, and the labels in the string format given in train.csv, and returns an image containing the bounding boxes and characters annotated

def visualize_training_data(image_fn, labels):

    # Convert annotation string to array

    labels = np.array(labels.split(' ')).reshape(-1, 5)

    #print(labels)

    

    # Read image

    imsource = Image.open(image_fn).convert('RGBA')

    bbox_canvas = Image.new('RGBA', imsource.size)

    char_canvas = Image.new('RGBA', imsource.size)

    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character

    char_draw = ImageDraw.Draw(char_canvas)



    for codepoint, x, y, w, h in labels:

        x, y, w, h = int(x), int(y), int(w), int(h)

        char = unicode_map[codepoint] # Convert codepoint to actual unicode character



        # Draw bounding box around character, and unicode character next to it

        bbox_draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))

        char_draw.text((x + w + fontsize/4, y + h/2 - fontsize), char, fill=(0, 0, 255, 255), font=font)

        Croped_image = imsource.crop((x, y, x+w, y+h))

        plt.figure()

        print(str(unicode_map[codepoint]))

        plt.imshow(Croped_image)

        plt.show()



    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)

    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

    return np.asarray(imsource)
np.random.seed(1337)



for i in range(1):

    img, labels = df_train.values[np.random.randint(len(df_train))]

    viz = visualize_training_data('../input/train_images/{}.jpg'.format(img), labels)

    

    plt.figure(figsize=(15, 15))

    plt.title(img)

    plt.imshow(viz, interpolation='lanczos')

    plt.show()
def preProcessImage(image):

    #image = np.asarray(image)

    #image = image.resize((300,300))

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret,th1 = cv2.threshold(image,155,255,cv2.THRESH_BINARY)

    return th1
# This function takes in a filename of an image, and the labels in the string format given in a submission csv, and returns an image with the characters and predictions annotated.

def Extract_Data():

    X_=[]

    y_=[]

    # Convert annotation string to array #300

    for img, labels in tqdm(df_train[:420].values):

        try:

            image_fn = '../input/train_images/{}.jpg'.format(img)

            labels = np.array(labels.split(' ')).reshape(-1, 5)

            # Read image

            imsource = Image.open(image_fn).convert('RGBA')

            bbox_canvas = Image.new('RGBA', imsource.size)

            char_canvas = Image.new('RGBA', imsource.size)

            bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character

            char_draw = ImageDraw.Draw(char_canvas)



            for codepoint, x, y, w, h in labels:

                x, y, w, h = int(x), int(y), int(w), int(h)

                char = unicode_map[codepoint] # Convert codepoint to actual unicode character



                # Draw bounding box around character, and unicode character next to it

                #bbox_draw.rectangle((x-10, y-10, x+10, y+10), fill=(255, 0, 0, 255))

                #char_draw.text((x+25, y-fontsize*(3/4)), char, fill=(255, 0, 0, 255), font=font)

                Croped_image = imsource.crop((x, y, x+w, y+h))

                image = Croped_image.resize((300,300))

                image = np.asarray(image)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                ret,th1 = cv2.threshold(image,155,255,cv2.THRESH_BINARY_INV)

                X_.append(th1)

                y_.append(str(unicode_map[codepoint]))

        except:

            pass

    X_ = np.array(X_)

    y_ = np.array(y_)



    '''imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)

    imsource = imsource.convert("RGB") '''# Remove alpha for saving in jpg format.

    return X_,y_
XX_,yy_ = Extract_Data()
plt.figure()

plt.imshow(XX_[99])
XX_.shape
unique, counts = np.unique(yy_, return_counts=True)

print(unique, counts )
NoOfClasses = len(unique)

NoOfClasses
IMG_ROWS=300

IMG_COLS=300

def PreProcessData(X,y):

    lb = LabelEncoder()

    y_integer = lb.fit_transform(y)

    out_y = np_utils.to_categorical(y_integer)

    num_images = X.shape[0]

    out_x = X.reshape(num_images, IMG_ROWS, IMG_COLS, 1)

    #out_x = x_shaped_array / 255

    return out_x, out_y
lb = LabelEncoder()

y_integer = lb.fit_transform(yy_)
X,y = PreProcessData(XX_,yy_)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)
K.clear_session()

def Kuzushiji_Classifier(in_):

    model_ = Conv2D(32,(3,3),activation='relu', padding="same")(in_)

    model_ = BatchNormalization()(model_)

    model_ =  Conv2D(32,(3, 3), activation='relu')(model_)

    model_ = BatchNormalization()(model_)

    model_ = Conv2D(32,5,strides=2,padding='same',activation='relu')(model_)

    model_ = MaxPooling2D((2, 2))(model_)

    model_ = BatchNormalization()(model_)

    model_ = Dropout(0.4)(model_)

    model_ = Conv2D(64,(3, 3), strides=2,padding='same', activation='relu')(model_)

    model_ = MaxPooling2D(pool_size=(2, 2))(model_)

    model_ = BatchNormalization()(model_)

    model_ = Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu')(model_)

    model_ = Dropout(0.4)(model_)

    model_ = Flatten()(model_)

    model_ = Dense(128, activation='relu')(model_)

    model_ = Dropout(0.4)(model_)

    model_ = Dense(NoOfClasses, activation='softmax')(model_)

    return model_
Input_Sample = Input(shape=(300, 300,1))

Output_ = Kuzushiji_Classifier(Input_Sample)

Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)
Model_Enhancer.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

Model_Enhancer.summary()
checkpointer = ModelCheckpoint('model_Kuzushiji.h5', verbose=0,mode='auto', monitor='val_acc',save_best_only=True)
ModelHistory = Model_Enhancer.fit(X_train, y_train,

          batch_size=100,

          epochs=32,

          verbose=1,callbacks=[checkpointer],

          validation_data=(X_val, y_val))
#Loss Curves

plt.figure(figsize=[20,9])

plt.plot(ModelHistory.history['loss'], 'r')

plt.plot(ModelHistory.history['val_loss'], 'b')

plt.legend(['Training Loss','Validation Loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Loss Curves')
#Accuracy Curves

plt.figure(figsize=[20,9])

plt.plot(ModelHistory.history['acc'], 'r')

plt.plot(ModelHistory.history['val_acc'], 'b')

plt.legend(['Training Accuracy','Validation Accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Accuracy Curves')
Model_ = load_model('model_Kuzushiji.h5')
def VisualizeKuzushiji(imagePath):

    img = cv2.imread(imagePath)

    imsource = Image.open(imagePath)#fromarray(img)

    char_draw = ImageDraw.Draw(imsource)

    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, im_th = cv2.threshold(im_grey, 130, 255, cv2.THRESH_BINARY_INV)

    ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    Kuzushijis = []

    for rect in rects:

        leng = int(rect[3] * 1.6)

        pt1 = int(rect[1] + rect[3]//2 - leng// 2)

        pt2 = int(rect[0] + rect[2]//2 - leng// 2)

        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

        #bbox_draw.rectangle((rect[0], rect[1], rect[0] + rect[2],rect[1] + rect[3]), fill=(0, 225, 0, 0))

        #print(roi.size)

        if roi.size>7000:

            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (225, 0, 0), 6)

            roi = cv2.resize(roi, (300,300))

            #roi = cv2.dilate(roi, (3, 3))

            ret,th1 = cv2.threshold(roi,155,255,cv2.THRESH_BINARY)

            ProcessImage = th1.reshape(1,IMG_ROWS, IMG_COLS, 1)

            y_pred = Model_.predict(ProcessImage)

            y_true = np.argmax(y_pred,axis=1)

            Kuzushiji = lb.inverse_transform(y_true)

            #print(Kuzushiji[0])

            Kuzushijis.append(str(Kuzushiji[0]))

            char_draw.text((rect[0]+10, rect[1]),str(Kuzushiji[0]), fill=(0,22,225,0), font=font)

            #cv2.putText(img, str(Kuzushiji[0]), (rect[0], rect[1]),font, 2, (0, 255, 255), 3)

    return img,imsource
img1, imsource1 = VisualizeKuzushiji('../input/train_images/100241706_00014_2.jpg')

plt.figure(figsize=(30,30))

plt.subplot(1,4,1)

plt.title("Detection of Kuzushiji",fontsize=20)

plt.imshow(img1)

plt.subplot(1,4,2)

plt.title("Recognition of Kuzushiji",fontsize=20)

plt.imshow(imsource1)
plt.figure(figsize=(30,30))

plt.title("Recognition of Kuzushiji",fontsize=20)

plt.imshow(imsource1)
img2, imsource2 = VisualizeKuzushiji('../input/test_images/test_001c37e2.jpg')

plt.figure(figsize=(30,30))

plt.subplot(1,4,1)

plt.title("Detection of Kuzushiji",fontsize=20)

plt.imshow(img2)

plt.subplot(1,4,2)

plt.title("Recognition of Kuzushiji",fontsize=20)

plt.imshow(imsource2)
plt.figure(figsize=(30,30))

plt.title("Recognition of Kuzushiji",fontsize=20)

plt.imshow(imsource2)
img3, imsource3 = VisualizeKuzushiji('../input/test_images/test_009f58c8.jpg')

plt.figure(figsize=(30,30))

plt.subplot(1,4,1)

plt.title("Detection of Kuzushiji",fontsize=20)

plt.imshow(img3)

plt.subplot(1,4,2)

plt.title("Recognition of Kuzushiji",fontsize=20)

plt.imshow(imsource3)
plt.figure(figsize=(30,30))

plt.title("Recognition of Kuzushiji",fontsize=20)

plt.imshow(imsource3)
img4, imsource4 = VisualizeKuzushiji('../input/test_images/test_1abdbbfe.jpg')

plt.figure(figsize=(30,30))

plt.subplot(1,4,1)

plt.title("Detection of Kuzushiji",fontsize=20)

plt.imshow(img4)

plt.subplot(1,4,2)

plt.title("Recognition of Kuzushiji",fontsize=20)

plt.imshow(imsource4)
plt.figure(figsize=(30,30))

plt.title("Recognition of Kuzushiji",fontsize=20)

plt.imshow(imsource4)