import numpy as np

import pandas as pd

import os

import cv2

from matplotlib import pyplot as plt

%matplotlib inline

from PIL import Image
#各種パス設定

BASEPATH = "../input/siim-isic-melanoma-classification"

JPEGPATH = "jpeg/train/"

df_train = pd.read_csv(os.path.join(BASEPATH, 'train.csv'))

df_test  = pd.read_csv(os.path.join(BASEPATH, 'test.csv'))

df_sub   = pd.read_csv(os.path.join(BASEPATH, 'sample_submission.csv'))
def cut_hokuro(img):

    kernel = np.ones((15,15),np.uint8)



    # クロージング(穴埋め)

    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations = 2)

    #15×15箱フィルタによる平均

    blur = cv2.blur(closing,(15,15))



    # グレスケ化

    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

    # 二値化

    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)





    # ほくろの輪郭情報取得

    contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    cnt = max(contours, key=cv2.contourArea)



    # 幅と高さ取得

    h, w = img.shape[:2]

    mask = np.zeros((h, w), np.uint8)



    # 輪郭の描画

    cv2.drawContours(mask, [cnt],-1, 255, -1)

    res = cv2.bitwise_and(img, img, mask=mask)

    

    # 対象を楕円で囲んだ時の中心点,長軸短軸,傾きを取得

    _,_,angle = cv2.fitEllipse(cnt)

    scale=1

    angle=angle+90

    

    return res,angle
base = []

convert = []

rotate = []

for i in range(0,50):

    img = cv2.imread(os.path.join(os.path.join(BASEPATH,JPEGPATH),df_train["image_name"][i]+".jpg"))

    img = cv2.resize(img,(256,256)) 

    base.append(img)

    #中心部分を抜き出して輪郭&角度取得

    cut_image,angle = cut_hokuro(img[64:192,64:192])

    #cut_image,angle = cut_hokuro(img[74:182,74:182])

    #cut_image,angle = cut_hokuro(img)

    convert.append(cut_image)

    rotate.append(angle)
i=1

fig = plt.figure(figsize=(10,10))

for base_image,hokuro_image in zip(base,convert):

    plt.subplot(10,10,i)

    base_image = cv2.cvtColor(base_image,cv2.COLOR_BGR2RGB)

    plt.imshow(base_image)

    plt.axis("off")

    plt.subplot(10,10,i+1)

    hokuro_image = cv2.cvtColor(hokuro_image,cv2.COLOR_BGR2RGB)

    plt.imshow(hokuro_image)

    i+=2

    plt.axis("off")
i=1

fig = plt.figure(figsize=(10,10))



#画像の幅と高さ

width = 256

height = 256



#画像の中心

center = (128,128)

#スケール

scale = 1



for base_image,angle in zip(base,rotate):

    base_image = cv2.cvtColor(base_image,cv2.COLOR_BGR2RGB)

    trans = cv2.getRotationMatrix2D(center,angle,scale)

    rotate_image = cv2.warpAffine(base_image,trans,(width,height))

    

    plt.subplot(10,10,i)

    plt.imshow(base_image)

    plt.axis("off")

    plt.subplot(10,10,i+1)

    plt.imshow(rotate_image)

    i+=2

    plt.axis("off")