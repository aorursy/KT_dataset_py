# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

i=1

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        print(i)

        i=i+1

# Any results you write to the current directory are saved as output.
#This is for listing directories in that folder

os.listdir('../input/cell_images/cell_images')  
img_list=[]

target=[]

for dirname in os.listdir("../input/cell_images/cell_images"):

    #print(dirname)

    for filename in os.listdir("../input/cell_images/cell_images/"+dirname):

        #print(filename)

        if filename!="Thumbs.db":

            img_list.append("../input/cell_images/cell_images/"+dirname+"/"+filename)

            target.append(dirname)

            

print(len(img_list),len(target))



data={'filename':img_list,'target':target}

df=pd.DataFrame(data)

print(df.head(7))

print(df.shape)

#print(df[:9])

print(df.filename[1])
from PIL import Image  #Python Imagine Library 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



img = Image.open(df.filename[1])

plt.imshow(img)

imgdata=[]

for i in df.filename:

    if i[-9:]!="Thumbs.db":

        img=Image.open(i)

        img1=img.resize((32,32))

        #img2=img1.convert("L")  #img2=img1.convert("LA")

        img3=np.array(img1)

        imgdata.append(img3)



imgdata=np.array(imgdata)        

print(len(imgdata))

imgdata.shape
print(imgdata[1].shape)

plt.imshow(imgdata[1])

imgdata[1]
print(df.target.shape)

print(imgdata.shape)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(imgdata,df.target,test_size=0.3,shuffle=True)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)