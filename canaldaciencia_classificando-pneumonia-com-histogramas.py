# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

pil_im = Image.open('../input/logocanal/LOGO PNG.png')

pil_im


from os import listdir

from os.path import isfile, join



mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'



pneumonia_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

import matplotlib.pyplot as plt

image=plt.imread(mypath+pneumonia_files[0])

plt.imshow(image)
import cv2

image=cv2.imread(mypath+pneumonia_files[0])

plt.imshow(image)
print(image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
laplacian = cv2.Laplacian(image,cv2.CV_64F)

plt.imshow(laplacian)
sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)

plt.imshow(sobelx)
sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

plt.imshow(sobely)
import cv2

import numpy as np



image=cv2.imread(mypath+pneumonia_files[0])

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,50,150,apertureSize = 3)



plt.imshow(edges)

plt.show()



lines = cv2.HoughLines(edges,1,np.pi/32,10)



for rho,theta in lines[0]:

    a = np.cos(theta)

    b = np.sin(theta)

    x0 = a*rho

    y0 = b*rho

    x1 = int(x0 + 1000*(-b))

    y1 = int(y0 + 1000*(a))

    x2 = int(x0 - 1000*(-b))

    y2 = int(y0 - 1000*(a))



    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),20)





plt.imshow(image)

plt.show()



! pip install scikit-fuzzy



import skfuzzy as fuzz
from skimage import transform,io

import cv2



def fuzzy(image):

    

    

    

    mfx = fuzz.trapmf(image.flatten(),  [0.1, 0.2,10,200])

    

 

    

    return mfx.reshape(image.shape[0], image.shape[1],3)
image=cv2.imread(mypath+pneumonia_files[0])

fuzzy_image = fuzzy(image)

plt.imshow(fuzzy_image)
fig, ax = plt.subplots(4, 2, figsize=(20, 20))

row=0

for file_path in (pneumonia_files):

    image = plt.imread(mypath+file_path)

    if row<4:

        ax[row, 0].imshow(image)

        ax[row, 1].hist(image.ravel(), 256, [0,256])

        ax[row, 0].axis('off')

    else:

        break

    row+=1

    if row == 0:

        ax[row, 0].set_title('Images')

        ax[row, 1].set_title('Histograms')

   



plt.show()
mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/'



normal_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

fig, ax = plt.subplots(4, 2, figsize=(20, 20))

row=0

for file_path in (normal_files):

    image = plt.imread(mypath+file_path)

    if row<4:

        ax[row, 0].imshow(image)

        ax[row, 1].hist(image.ravel(), 256, [0,256])

        ax[row, 0].axis('off')

    else:

        break

    row+=1

    if row == 0:

        ax[row, 0].set_title('Images')

        ax[row, 1].set_title('Histograms')

   



plt.show()
len(normal_files)
len(pneumonia_files)
histogramas_normais=[]

mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/'



y_normal=[]



for file_path in (normal_files[:200]):

    image =cv2.imread(mypath+file_path)

    histogram=cv2. cv2.calcHist([image],      # image

                               [0, 1],           # channels

                               None,             # no mask

                               [180, 256],       # size of histogram

                               [0, 180, 0, 256]  # channel values

                               )



    histogramas_normais.append( histogram.ravel())

    y_normal.append(0)
histogramas_pneumonia=[]

mypath='/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'





y_pneumonia=[]



for file_path in (pneumonia_files[:200]):

    image =cv2.imread(mypath+file_path)

    histogram=cv2. cv2.calcHist([image],      # image

                               [0, 1],           # channels

                               None,             # no mask

                               [180, 256],       # size of histogram

                               [0, 180, 0, 256]  # channel values

                               )



    histogramas_pneumonia.append( histogram.ravel())

    y_pneumonia.append(1)
y=y_normal+y_pneumonia
X=histogramas_normais+histogramas_pneumonia
df=pd.DataFrame(X)
df['y']=y
df=df.sample(frac=1)
X=df.drop(['y'],axis=1)
y=df['y']
from sklearn import tree

clf = tree.DecisionTreeClassifier()

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_test, y_pred)
from sklearn.metrics import f1_score

f1_score(y_test, y_pred)
import pickle



filename = '/kaggle/working/pneumonia_histograma.sav'

pickle.dump(clf, open(filename, 'wb'))