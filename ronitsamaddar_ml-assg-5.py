# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/flowers/flowers/sunflower"))

# Any results you write to the current directory are saved as output.
import skimage.transform
import skimage.io
Files=[]
Flowers=os.listdir("../input/flowers/flowers")
print(Flowers)
for fl in Flowers:
    Files.append(os.listdir("../input/flowers/flowers/"+fl))
print(len(Files))
print(len(Files[0]))
flow=Flowers[0]
file=Files[0][2]
img=plt.imread("../input/flowers/flowers/"+flow+"/"+file)
plt.imshow(img)
print("HELLO1")
img1=plt.imread("../input/flowers/flowers/"+Flowers[0]+"/"+Files[0][0])
imgSet=np.reshape(skimage.transform.resize(img1,[120,120]),(1,120,120,3))
print("HELLO2")
ySet=[Flowers[0]]
for i in range(len(Flowers)):
    for j in range(len(Files[i])):
        if(i!=0 or j!=0):
            try:
                print(i,j)
                x=plt.imread("../input/flowers/flowers/"+Flowers[i]+"/"+Files[i][j])
                x=np.reshape(skimage.transform.resize(x,[120,120]),(1,120,120,3))
                imgSet=np.append(imgSet,x,axis=0)
                #print(x.shape)
                ySet.append(Flowers[i])
            except:
                continue
print("DONE!!!!!!!")

    



np.save('imgSet.npy',imgSet)
np.save('ySet.npy',ySet)





