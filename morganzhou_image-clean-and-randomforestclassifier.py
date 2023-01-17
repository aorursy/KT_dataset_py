# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

import time



#model selection

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2                  



# to make this notebook's output stable across runs

np.random.seed(42)
IMG_SIZE = 128

DATA_DIR = os.path.join(r'/kaggle/input/', 'aptos2019-blindness-detection')

TRAIN_DIR = os.path.join(DATA_DIR,'train_images')

TEST_DIR = os.path.join(DATA_DIR, 'test_images')



def searchCircle(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 5)

    rows = gray.shape[0]

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,

                               param1=100, param2=30,

                               minRadius=int(img.shape[1]/4), maxRadius=int(img.shape[1]/2))

    

    if circles is not None:

        circles = np.int32(np.around(circles))

        return circles[0,0]

    

    cx = int(img.shape[1]/2)

    cy = int(img.shape[0]/2)

    r = max(cx, cy)

    return np.array([cx, cy, r])



def circle_crop(img, cx, cy, r):

    mask = np.zeros((img.shape[1], img.shape[0]), np.uint8)

    cv2.circle(mask, (cx, cy), r, (255,255,255), thickness=-1)

    return cv2.bitwise_and(img, img, mask=mask)



def rect_crop(img, x0, y0, x1, y1):

    mask = np.zeros((img.shape[1], img.shape[0]), np.uint8)

    cv2.rectangle(mask, (x0, y0), (x1, y1), (255,255,255), thickness=-1)

    return cv2.bitwise_and(img, img, mask=mask)



def processImg(img):

    # find the circle and resize

    cir = searchCircle(img)     

    scale = (float(IMG_SIZE)) / (cir[2] * 2)

    img = cv2.resize(img,(0,0),fx=scale, fy=scale)

    

    cir = np.int32(np.around(cir * scale))

    

    # crop the circle

    x0 = max(cir[0]-cir[2], 0)

    x1 = min(cir[0]+cir[2], img.shape[1])

    y0 = max(cir[1]-cir[2], 0)

    y1 = min(cir[1]+cir[2], img.shape[0])

    img = img[y0:y1, x0:x1,:]

    

    # copy the circle to the center of square

    img1 = np.zeros([cir[2] * 2, cir[2] * 2, img.shape[2]])

    dx = max(cir[2] - cir[0], 0)

    dy = max(cir[2] - cir[1], 0)

    img1[dy:dy+y1-y0,dx:dx+x1-x0,:] = img

    

    # crop the cirle

    r = int(IMG_SIZE/2)

    img1 = circle_crop(img1, r, r, r)

    img1 = rect_crop(img1, dx, dy, dx+x1-x0, dy+y1-y0)



    img1 = np.array(img1)/255 # normalize the image

    return img1



def getImageData(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)



    if img is not None:

        return processImg(img)

    

    return None



def idcode2Path(imgDir, idCode):

    return os.path.join(imgDir, '{}.png'.format(idCode))



def getImgArray(idList, imgDir):

    imgs = []

    for i, idCode in enumerate(idList):

        img = getImageData(idcode2Path(imgDir, idCode))

        if img is None:

            print(idCode , " is none")

        imgs.append(img)

        

    return np.array(imgs)



def showOne(ax, imgs, index):

    if(index >= len(imgs)): 

        return False

    

    img1 = imgs[index].reshape(IMG_SIZE, IMG_SIZE, 3)

    ax.imshow(img1)

    ax.axis("off")

    

    return True



def plotMatrix(funcOne, paramList, row=1, col=3, figsize=None):

    if figsize is None:

        figsize=(col * 5, row * 4)

    fig, axes = plt.subplots(row, col,figsize=figsize)

    if((row > 1) | (col > 1)):

        axes = axes.ravel()

    else:

        axes = [axes]

        

    for i, ax in enumerate(axes):

        if(not funcOne(ax, paramList, i)): 

            break

   

    plt.show()
df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

print(df.shape)
t0 = time.time()

X = getImgArray(df['id_code'], TRAIN_DIR)

X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])

Y = df['diagnosis']



print("getImgArray elapsed time:", time.time() - t0)
plotMatrix(showOne, X, 2, 3)
x_train,x_valid,y_train,y_valid = train_test_split(X,Y,test_size=0.2,random_state=42)
model = RandomForestClassifier(random_state=42, bootstrap=False, criterion='entropy', 

           max_features=10,min_samples_split=10, max_depth=38, n_estimators=27)



model.fit(x_train, y_train)



y_valid_pred = model.predict(x_valid)



score = accuracy_score(y_valid, y_valid_pred)

print("Valid Score = {0:.4f}".format(score))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

x_test = getImgArray(test_df['id_code'], TEST_DIR)



print(x_test.shape)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
model.fit(X, np.array(Y))

pred = model.predict(x_test)



test_df['diagnosis'] = pred

test_df.to_csv('submission.csv',index=False)