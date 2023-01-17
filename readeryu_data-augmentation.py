# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import cv2

princess = cv2.imread("../input/.jpg")
import cv2

import numpy as np

import matplotlib.pyplot as plt 

l,w=princess.shape[1],princess.shape[0]

def show_bgr_image(image,title='image',show=True,re=False):

    rgb = image[:,:,[2,1,0]]

    if show:

        plt.title(title)

        plt.imshow(rgb)

    if re:

        return rgb

    

##图像变成指定大小##

resized_image = cv2.resize(princess,(1200,1200))



##对样本相关阵进行特征值分解

def color_augmention(img,sigma=0.1):

    image = img.reshape(-1, 3).astype(np.float32)

    scaling_factor = np.sqrt(3.0 / np.sum(np.var(image, axis=0)))

    image *= scaling_factor

    cov = np.cov(image.T)

    e,v = np.linalg.eig(cov)

    alpha = np.random.randn(1,3)*sigma

    print('扰动后的特征值大小',alpha*e)

    delta = v.dot((alpha*e).T)

    delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(img+delta.reshape(1,1,3), 0, 255).astype(np.uint8)

    return img_out

figure_color = plt.figure(figsize=(10,10))

##看九次随机跑出来的图片内容

for i in range(3):

    for j in range(3):

        plt.subplot(3,3,3*i+j+1)

        show_bgr_image(color_augmention(resized_image),title='')

plt.show()