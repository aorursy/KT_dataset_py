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

from keras.models import Model,load_model

print(os.listdir("../input/"))
fontsize = 50



# From https://www.google.com/get/noto/

!wget -q --show-progress https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip

!unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf > NotoSansCJKjp-Regular.otf

!rm NotoSansCJKjp-hinted.zip



font = ImageFont.truetype('./NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')
df_train = pd.read_csv('../input/kuzushiji-recognition/train.csv')

unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../input/kuzushiji-recognition/unicode_translation.csv').values}

unicode_map
reversed_unicode_map = dict(map(reversed, unicode_map.items()))

reversed_unicode_map
reversed_unicode_map['え']
# This function takes in a filename of an image, and the labels in the string format given in a submission csv, and returns an image with the characters and predictions annotated.

def Extract_Data():

    X_=[]

    y_=[]

    # Convert annotation string to array #300

    for img, labels in tqdm(df_train[:420].values):

        try:

            image_fn = '../input/kuzushiji-recognition/train_images/{}.jpg'.format(img)

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
Model_ = load_model('../input/kuzushijirecognitionweight/model_Kuzushiji.h5')
IMG_ROWS=300

IMG_COLS=300

def SubmissionKuzushiji(imagePath):

    img = cv2.imread(imagePath)

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

        if roi.size>7000:

            roi = cv2.resize(roi, (300,300))

            ret,th1 = cv2.threshold(roi,155,255,cv2.THRESH_BINARY)

            ProcessImage = th1.reshape(1,IMG_ROWS, IMG_COLS, 1)

            y_pred = Model_.predict(ProcessImage)

            y_true = np.argmax(y_pred,axis=1)

            Kuzushiji = lb.inverse_transform(y_true)

            Unicode_kuzushiji = reversed_unicode_map[str(Kuzushiji[0])]

            #print(Kuzushiji[0])

            Kuzushijis.append(Unicode_kuzushiji+" "+str(rect[0])+" "+str(rect[1]))

    result=' '.join(Kuzushijis)

    return result
results = SubmissionKuzushiji('../input/kuzushiji-recognition/test_images/test_001c37e2.jpg')

results
Images = sorted(os.listdir("../input/kuzushiji-recognition/test_images"))
len(Images)
labels=[]

image_id=[]

#print(os.listdir("../input/kuzushiji-recognition/test_images"))

for img in tqdm(Images[:830]):

    recognition = SubmissionKuzushiji('../input/kuzushiji-recognition/test_images/'+img)

    image_id.append(img[:-4])

    labels.append(recognition)

for img in tqdm(Images[830:1660]):

    recognition = SubmissionKuzushiji('../input/kuzushiji-recognition/test_images/'+img)

    image_id.append(img[:-4])

    labels.append(recognition)

for img in tqdm(Images[1660:2490]):

    recognition = SubmissionKuzushiji('../input/kuzushiji-recognition/test_images/'+img)

    image_id.append(img[:-4])

    labels.append(recognition)

for img in tqdm(Images[2490:3320]):

    recognition = SubmissionKuzushiji('../input/kuzushiji-recognition/test_images/'+img)

    image_id.append(img[:-4])

    labels.append(recognition)

for img in tqdm(Images[3320:4150]):

    recognition = SubmissionKuzushiji('../input/kuzushiji-recognition/test_images/'+img)

    image_id.append(img[:-4])

    labels.append(recognition)
len(labels)
my_submission = pd.DataFrame({'image_id': image_id, 'labels': labels})

my_submission.to_csv('SubmissionVictorKuzushiji.csv', index=False)
my_submission.head()