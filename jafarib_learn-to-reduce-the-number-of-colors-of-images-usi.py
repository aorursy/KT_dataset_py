import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from PIL import Image
img = np.array(Image.open('/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/rainbow_rect.jpg'))
print(img.dtype)

def im2double(im):

    min_val = np.min(im.ravel())

    max_val = np.max(im.ravel())

    out = (im.astype('float') - min_val) / (max_val - min_val)

    return out
img = im2double(img)
print(img.dtype)

img
import matplotlib.pyplot 
matplotlib.pyplot.imshow(img)
R = img[:,:,0]

G = img[:,:,1]

B = img[:,:,2]

R.shape
R
np.amax(R)
matplotlib.pyplot.imshow(R)
matplotlib.pyplot.imshow(G)
matplotlib.pyplot.imshow(B)
R.shape
img.shape
R.flatten().shape
  

np.reshape(R, R.flatten().shape)
X = np.transpose(np.array([R.flatten() , G.flatten() ,B.flatten() ]))
X
X.shape
np.savetxt("RGB.csv", X, delimiter=",")
np.amax(X)
from sklearn.cluster import KMeans
k=25
kmeans = KMeans(n_clusters=k,random_state=0 ).fit(X)
IDX = kmeans.labels_

IDX 
IDX.max()
np.savetxt("IDX.csv", IDX, delimiter=",")
C = kmeans.cluster_centers_

C
C.shape
X2 = C[IDX,:]

X2
X2.shape
R2=np.reshape(X2[:,0], R.shape)

G2=np.reshape(X2[:,1], G.shape)

B2=np.reshape(X2[:,2], B.shape)

R2
R2.shape
img2 = np.zeros(img.shape)

img2.shape
img2[:,:,0]= R2

img2[:,:,1]= G2

img2[:,:,2]= B2

img2
matplotlib.pyplot.imshow(img2)
fig=matplotlib.pyplot.figure(figsize = (15,7))

fig.add_subplot(1,2,1)

matplotlib.pyplot.imshow(img,aspect='auto')

fig.add_subplot(1,2,2)

matplotlib.pyplot.imshow(img2,aspect='auto')

 
k=50

kmeans = KMeans(n_clusters=k,random_state=0 ).fit(X)

IDX = kmeans.labels_

C = kmeans.cluster_centers_

X3 = C[IDX,:]

R3=np.reshape(X3[:,0], R.shape)

G3=np.reshape(X3[:,1], G.shape)

B3=np.reshape(X3[:,2], B.shape)

img3 = np.zeros(img.shape)

img3[:,:,0]= R3

img3[:,:,1]= G3

img3[:,:,2]= B3

fig=matplotlib.pyplot.figure(figsize = (18,7))

fig.add_subplot(1,3,1)

matplotlib.pyplot.imshow(img,aspect='auto')

fig.add_subplot(1,3,2)

matplotlib.pyplot.imshow(img2,aspect='auto')

fig.add_subplot(1,3,3)

matplotlib.pyplot.imshow(img3,aspect='auto')
def color_reduction_withKmeans (imgaddress, k):

    imgage = np.array(Image.open(imgaddress))

    img = im2double(imgage)

    R = img[:,:,0]

    G = img[:,:,1]

    B = img[:,:,2]

    X = np.transpose(np.array([R.flatten() , G.flatten() ,B.flatten() ]))

    kmeans = KMeans(n_clusters=k,random_state=0 ).fit(X)

    IDX = kmeans.labels_

    C = kmeans.cluster_centers_

    X2 = C[IDX,:]

    R2=np.reshape(X2[:,0], R.shape)

    G2=np.reshape(X2[:,1], G.shape)

    B2=np.reshape(X2[:,2], B.shape)

    img2 = np.zeros(img.shape)

    img2[:,:,0]= R2

    img2[:,:,1]= G2

    img2[:,:,2]= B2

    fig=matplotlib.pyplot.figure(figsize = (15,7))

    fig.add_subplot(1,2,1)

    matplotlib.pyplot.imshow(img,aspect='auto')

    fig.add_subplot(1,2,2)

    matplotlib.pyplot.imshow(img2,aspect='auto')

    

    

    
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/rainbow_rect.jpg'

k=25

color_reduction_withKmeans (imgaddress, k)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/rainbow_rect.jpg'

k=50

color_reduction_withKmeans (imgaddress, k)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/maghbaratoshoara_tabriz.jpeg'

k=50

color_reduction_withKmeans (imgaddress, k)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/maghbaratoshoara_tabriz.jpeg'

k=25

color_reduction_withKmeans (imgaddress, k)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/maghbaratoshoara_tabriz.jpeg'

k=10

color_reduction_withKmeans (imgaddress, k)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/maghbaratoshoara_tabriz.jpeg'

k=5

color_reduction_withKmeans (imgaddress, k)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/maghbaratoshoara_tabriz.jpeg'

k=3

color_reduction_withKmeans (imgaddress, k)
import colorsys

def convert_RGB_Matrice_to_HSV(X):

    Y =  np.zeros(X.shape)

    for i in range(len(X)):

        Z = colorsys.rgb_to_hsv(X[i][0],X[i][1],X[i][2])

        Y[i][0] =Z[0]

        Y[i][1] =Z[1]

        Y[i][2] =Z[2]

        

    return Y
def convert_HSV_Matrice_to_RGB(Y):

    X =  np.zeros(Y.shape)

    for i in range(len(Y)):

        Z = colorsys.hsv_to_rgb(Y[i][0],Y[i][1],Y[i][2])

        X[i][0] =Z[0]

        X[i][1] =Z[1]

        X[i][2] =Z[2]

        

    return X


def color_reduction_withKmeans_RGB2HSV (imgaddress, k):

    imgage = np.array(Image.open(imgaddress))

    img = im2double(imgage)

    R = img[:,:,0]

    G = img[:,:,1]

    B = img[:,:,2]

    X = np.transpose(np.array([R.flatten() , G.flatten() ,B.flatten() ]))

    Y = convert_RGB_Matrice_to_HSV(X)

    kmeans = KMeans(n_clusters=k,random_state=0 ).fit(Y)

    IDX = kmeans.labels_

    C = kmeans.cluster_centers_

    Y2= C[IDX,:]

    X2 = convert_HSV_Matrice_to_RGB(Y2)

    R2=np.reshape(X2[:,0], R.shape)

    G2=np.reshape(X2[:,1], G.shape)

    B2=np.reshape(X2[:,2], B.shape)

    img2 = np.zeros(img.shape)

    img2[:,:,0]= R2

    img2[:,:,1]= G2

    img2[:,:,2]= B2

    fig=matplotlib.pyplot.figure(figsize = (15,7))

    fig.add_subplot(1,2,1)

    matplotlib.pyplot.imshow(img,aspect='auto')

    fig.add_subplot(1,2,2)

    matplotlib.pyplot.imshow(img2,aspect='auto')

    

    
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/maghbaratoshoara_tabriz.jpeg'

k=20

color_reduction_withKmeans_RGB2HSV (imgaddress, k)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/rainbow_rect.jpg'

k=20

color_reduction_withKmeans_RGB2HSV (imgaddress, k)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/rainbow_rect.jpg'

k=30

color_reduction_withKmeans_RGB2HSV (imgaddress, k)
def color_reduction_withKmeans_RGB2HSV_with_whight (imgaddress, k,hw,sw,vw):

    imgage = np.array(Image.open(imgaddress))

    img = im2double(imgage)

    R = img[:,:,0]

    G = img[:,:,1]

    B = img[:,:,2]

    X = np.transpose(np.array([R.flatten() , G.flatten() ,B.flatten() ]))

    Y = convert_RGB_Matrice_to_HSV(X)

    W = [hw,sw,vw]

    Y[:,0]=W[0]* Y[:,0]

    Y[:,1]=W[1]* Y[:,1]

    Y[:,2]=W[2]* Y[:,2]

    

    kmeans = KMeans(n_clusters=k,random_state=0 ).fit(Y)

    IDX = kmeans.labels_

    C = kmeans.cluster_centers_

    Y2= C[IDX,:]

    Y2[:,0]=Y2[:,0]/W[0]  

    Y2[:,1]=Y2[:,1]/W[1]  

    Y2[:,2]=Y2[:,2]/W[2]  

    X2 = convert_HSV_Matrice_to_RGB(Y2)

    R2=np.reshape(X2[:,0], R.shape)

    G2=np.reshape(X2[:,1], G.shape)

    B2=np.reshape(X2[:,2], B.shape)

    img2 = np.zeros(img.shape)

    img2[:,:,0]= R2

    img2[:,:,1]= G2

    img2[:,:,2]= B2

    fig=matplotlib.pyplot.figure(figsize = (15,7))

    fig.add_subplot(1,2,1)

    matplotlib.pyplot.imshow(img,aspect='auto')

    fig.add_subplot(1,2,2)

    matplotlib.pyplot.imshow(img2,aspect='auto')
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/rainbow_rect.jpg'

k=30

hw=5

sw=1

vw=2

color_reduction_withKmeans_RGB2HSV_with_whight (imgaddress, k,hw,sw,vw)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/rainbow_rect.jpg'

k=30

hw=0.1

sw=1

vw=2

color_reduction_withKmeans_RGB2HSV_with_whight (imgaddress, k,hw,sw,vw)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/rainbow_rect.jpg'

k=20

hw=40

sw=1

vw=2

color_reduction_withKmeans_RGB2HSV_with_whight (imgaddress, k,hw,sw,vw)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/rainbow_rect.jpg'

k=20

hw=5

sw=1

vw=3

color_reduction_withKmeans_RGB2HSV_with_whight (imgaddress, k,hw,sw,vw)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/maghbaratoshoara_tabriz.jpeg'

k=20

hw=5

sw=1

vw=3

color_reduction_withKmeans_RGB2HSV_with_whight (imgaddress, k,hw,sw,vw)
imgaddress = '/kaggle/input/reduce-the-number-of-colors-unsupervised-learning/maghbaratoshoara_tabriz.jpeg'

k=20

hw=4

sw=1

vw=3

color_reduction_withKmeans_RGB2HSV_with_whight (imgaddress, k,hw,sw,vw)
!pip install -U scikit-fuzzy

import skfuzzy as fuzz

!pip install -U  fuzzy-c-means

from fcmeans import FCM
K=30

fcm = FCM(n_clusters=K)

fcm.fit(X)
# outputs

C = fcm.centers

U  = fcm.u 
U.shape
M=np.amax(U,axis=1)
M.shape
M
M.transpose()