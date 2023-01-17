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
        pass
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import PIL
from PIL import Image
import matplotlib.pyplot as plt

plt.figure(figsize = (10,15))
for i in range(1,8):
    plt.subplot(4,4,i)
    try:
        img = Image.open("/kaggle/input/cookie/kaggledataset/" +str(i)+ "/cookie"+str(i)+".jpg")
    except:
        pass
    plt.imshow(img)
def remove_noise(image):
    return cv2.medianBlur(image,3)


def threshold_func(data_array):
        
    data_array = data_array.reshape(1,2500)

    threshold = data_array.sum()/2500
    threshold = (threshold*50)/100
    
    data_array = data_array.reshape(50,50)
    for i in range(50):
        for j in range(50):
            if data_array[i][j] < threshold:
                data_array[i][j] = 0
            else:
                data_array[i][j] = 255
                
    return data_array
import cv2
plt.figure(figsize = (10,15))
for i in range(1,9):
    plt.subplot(4,4,i)
    
    img = cv2.imread("/kaggle/input/cookie/kaggledataset/" +str(i)+ "/cookie"+str(i)+".jpg",0)
    img = cv2.resize(img, (50,50))
    img = threshold_func(img)
    #img = remove_noise(img)

    try:
        plt.imshow(img, cmap = 'gray')
    except:
        pass
import cv2
plt.figure(figsize = (10,15))
for i in range(1,9):
    plt.subplot(4,4,i)
    
    img = cv2.imread("/kaggle/input/cookie/kaggledataset/" +str(i)+ "/cookie"+str(i)+".jpg",0)
    img = cv2.resize(img, (50,50))
    img = threshold_func(img)
    img = remove_noise(img)

    try:
        plt.imshow(img, cmap = 'gray')
    except:
        pass
