# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import skimage.transform
import skimage.io



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/flowers/flowers"))
print(os.listdir("../input/flowers/flowers/sunflower"))

# Any results you write to the current directory are saved as output.
list=os.listdir("../input/flowers/flowers")
print(list)

    

list1=[]
for flower in list:
    list1.append(os.listdir("../input/flowers/flowers/"+flower))
print(list1)

import matplotlib.pyplot as plt
X=plt.imread("../input/flowers/flowers/"+list[0]+"/"+list1[0][0],format=None)
plt.imshow(X)

XSet=[]
YSet=[]
for i in range(0,len(list)):
    for j in range(0,len(list1[i])):
           # print("I="+str(i)+"J="+str(j))
            
           # print("Hello")
        try:
            x=plt.imread("../input/flowers/flowers/"+str(list[i])+"/"+str(list1[i][j]))
            x=np.reshape(skimage.transform.resize(x,[120,120]),(120,120,3))
            #print(x)
            XSet.append(x)
            YSet.append(list[i])
        except:
            continue
            
    
    
print("X.shape : "+str(len(XSet)))
print("Y.shape : "+str(len(YSet)))

                

        
print(len(XSet[0]))
XSet=np.array(XSet)
YSet=np.array(YSet)
print(XSet.shape)
print(YSet.shape)

np.save('imgSet.npy',XSet)
np.save('ySet.npy',YSet)



