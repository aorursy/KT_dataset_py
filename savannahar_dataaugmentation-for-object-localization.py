import numpy as np 

import pandas as pd 

import os



print(os.listdir("/kaggle/input/repository/savannahar68-DataAugmentation-dce458d/"))

PATH = "/kaggle/input/repository/savannahar68-DataAugmentation-dce458d/"

df = pd.read_csv(PATH + 'dataAug.csv')

df.head(5)
df = df.drop(df.columns[0], axis=1)
df.shape
import numpy as np, os, cv2

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from PIL import Image

%matplotlib inline
df.values
images = df.values

img_name = images[0][0]

x1 = images[0][1]

x2 = images[0][2]

y1 = images[0][3]

y2 = images[0][4]
plt.imshow(Image.open(PATH + 'images/' + img_name))

plt.gca().add_patch(Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none'))
x11 = 640 - x1

x22 = 640 - x2

y11 = y1

y22 = y2

plt.imshow(cv2.flip(cv2.imread(PATH + "images/"+ img_name),1))

plt.gca().add_patch(Rectangle((x11,y11),x22-x11,y22-y11,linewidth=1,edgecolor='r',facecolor='none'))
x11 = x1

x22 = x2

y11 = 480 - y1

y22 = 480 - y2

plt.imshow(cv2.flip(cv2.imread(PATH + "images/"+ img_name),0))

plt.gca().add_patch(Rectangle((x11,y11),x22-x11,y22-y11,linewidth=1,edgecolor='r',facecolor='none'))
x11 = 640 - x1

x22 = 640 - x2

y11 = 480 - y1

y22 = 480 - y2

plt.imshow(cv2.flip(cv2.imread(PATH + "images/"+ img_name),-1))

plt.gca().add_patch(Rectangle((x11,y11),x22-x11,y22-y11,linewidth=1,edgecolor='r',facecolor='none'))
for i,img in enumerate(df.values):

    x1 = img[1]

    x2 = img[2]

    y1 = img[3]

    y2 = img[4]

    

    df = df.append({'image_name':img[0].split('.')[0]+'H_FLIP.png', 'x1':640-x1, 'x2':640-x2, 'y1':y1, 'y2':y2}, ignore_index=True)

    cv2.imwrite(PATH + "images/" + img[0].split('.')[0]+'H_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0]),1))

    

    df = df.append({'image_name':img[0].split('.')[0]+'V_FLIP.png', 'x1':x1, 'x2':x2, 'y1':480-y1, 'y2':480-y2}, ignore_index=True)

    cv2.imwrite(PATH + "images/"+img[0].split('.')[0]+'V_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0]),0)) 

    

    df = df.append({'image_name':img[0].split('.')[0]+'H_FLIPV_FLIP.png', 'x1':640-x1, 'x2':640-x2, 'y1':480-y1, 'y2':480-y2}, ignore_index=True)

    cv2.imwrite(PATH + "images/"+img[0].split('.')[0]+'H_FLIPV_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0].split('.')[0]+'H_FLIP.png'),0))

   
df.shape
df.head(20) #This we have successfully augmented the data for bouding box prediction
for i,img in enumerate(df.values):

    bx = img[1]

    by = img[2]

    bh = img[3]

    bw = img[4]

    

    df = df.append({'image_name':img[0].split('.')[0]+'H_FLIP.png', 'bx':640 - bx, 'by':by, 'bh':bh, 'bw':bw}, ignore_index=True)

    cv2.imwrite(PATH + "images/" + img[0].split('.')[0]+'H_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0]),1))

    

    df = df.append({'image_name':img[0].split('.')[0]+'V_FLIP.png', 'bx':bx, 'by':480 - by, 'bh':bh, 'bw':bw}, ignore_index=True)

    cv2.imwrite(PATH + "images/"+img[0].split('.')[0]+'V_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0]),0)) 

    

    df = df.append({'image_name':img[0].split('.')[0]+'H_FLIPV_FLIP.png', 'bx':640 - bx, 'by':640-by, 'bh':bh, 'bw':bw}, ignore_index=True)

    cv2.imwrite(PATH + "images/"+img[0].split('.')[0]+'H_FLIPV_FLIP.png', cv2.flip(cv2.imread(PATH+"images/"+img[0].split('.')[0]+'H_FLIP.png'),0))

   