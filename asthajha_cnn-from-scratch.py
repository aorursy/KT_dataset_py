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

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt 

import cv2
def drawImg(img,title="Image"):

    plt.imshow(img,cmap="gray")  

    plt.axis("off")   #

    plt.style.use("seaborn")

    plt.title(title+str(img.shape))

    plt.show()

    
img_=cv2.imread('/kaggle/input/image-cnnbasic/shaktiman.jpg')   #bgr

img_=cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)   #for conversion

#plt.imshow(img_)

#img_=cv2.resize(img_,(100,100))         #for easy computatipn

img_gray = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

drawImg(img_)

drawImg(img_gray)
def convolution(img,img_filter):

    W=img.shape[0]

    H=img.shape[1]

    F=img_filter.shape[0]  #3

    new_img=np.zeros((W-F+1,H-F+1))

    

    for row in range(W-F+1):

        for col in range(H-F+1):

            for i in range(F):

                for j in range(F):

                    new_img[row][col]+=img[row+i][col+j]*img_filter[i][j]

                    

                if new_img[row][col]>255:

                    new_img[row][col]=255

                elif new_img[row][col]<0:

                    new_img[row][col]=0

                    

    return new_img  #activation map

                    
blur_filter =np.ones((3,3))/9.0



# print(blur_filter)

output1=convolution(img_gray,blur_filter)
drawImg(img_gray)

drawImg(output1)
edge_filter= np.array([[1,0,-1],

                       [1,0,-1],

                       [1,0,-1]])

output2=convolution(img_gray,edge_filter)

drawImg(img_gray)

drawImg(output2)
print(img_.shape)

drawImg(img_)
#Pad using numpy 



pad_img=np.pad(img_,((20,10),(10,20),(0,0)),'constant',constant_values=0)

drawImg(pad_img)
X=np.array([[1,0,2,3],

            [4,6,6,8],

            [3,1,1,0],

            [1,2,2,4]])



def pooling(X,mode="max"):

    stride=2

    f=2

    H,W =X.shape

    

    HO=int((H-f)/stride)+1

    WO=int((W-f)/stride)+1

    

    output=np.zeros((HO,WO))

    

    for r in range(HO):

        for c in range(WO):

            r_start=r*stride

            r_end=r_start+f

            c_start=c*stride

            c_end=c_start+f

            

            X_slice= X[r_start:r_end,c_start:c_end]

            

            

            if mode=="max":

                

                output[r][c]=np.max(X_slice)

            else:

                output[r][c]=np.mean(X_slice)

                

    return output

            

            

    
pooling_output=pooling(X)

print(pooling_output)