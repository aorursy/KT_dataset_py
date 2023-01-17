import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img = cv2.imread("../input/sample3/images.png")
plt.imshow(img)
plt.show()
print("Image shape(Color image have 3 channel)",img.shape)
img1=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img1=np.float32(img1)
print("Image shape(Gray image have 1 channel)",img1.shape)
plt.imshow(img1)
plt.show()
print(img1)
sobel_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
sobel_x=sobel_y.T
print("sobel_x")
print("-----------")
print(sobel_x)

print("sobel)y")
print("-----------")
print(sobel_y)
Ix=cv2.filter2D(img1,-1,sobel_x)
Iy=cv2.filter2D(img1,-1,sobel_y)
Ixx=cv2.GaussianBlur(Ix*Ix,(3,3),1)
Iyy=cv2.GaussianBlur(Iy*Iy,(3,3),1)
Ixy=cv2.GaussianBlur(Ix*Iy,(3,3),1)
detM=Ixx*Iyy-(Ixy)**2
print("detM :- ",detM)
traceM=Ixx+Iyy
print("TraceM :- ",traceM)
k=0.05
harris_response=detM-k*(traceM)**2
img_copyforedge=np.copy(img)
img_copyforcorner=np.copy(img)
for rowindex , response in enumerate(harris_response):
    for colindex , r in enumerate(response):
        if r>100000:
            img_copyforcorner[rowindex,colindex]=[255,0,0]
        elif r<0:
            img_copyforedge[rowindex,colindex]=[0,255,0]
fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(15,15))
ax[0].set_title("Edges Found")
ax[0].imshow(img_copyforedge)
ax[1].set_title("Corners Found")
ax[1].imshow(img_copyforcorner)
ax[2].set_title("Orginal Image")
ax[2].imshow(img)
plt.show()
#plt.imshow(img)
