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
import cv2

import matplotlib.pyplot as plt

import numpy as np



image = cv2.imread('/kaggle/input/hand-written-digits/unnamed.jpg')

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

plt.subplot(1, 2, 1)

plt.title("Original")

plt.imshow(image)



grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

_, threshed_img = cv2.threshold(grey, 90, 255, cv2.THRESH_BINARY)

inv = cv2.bitwise_not(threshed_img)



plt.subplot(1, 2, 2)

plt.title("Grey")

plt.imshow(inv)
struct = np.ones((3,3),np.uint8)

dilated = cv2.dilate(inv ,struct,iterations=1)

edges = cv2.Canny(dilated,30,200)



plt.title("edges")

plt.imshow(edges)
contours, hier = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

contours.sort(key=lambda x:cv2.boundingRect(x)[0])



print('-- Contours sorted')
results=[]



for i,cnt in enumerate(contours):

    x,y,w,h = cv2.boundingRect(cnt)

    small_image = inv[y:y+h,x:x+w]

    small_image = np.pad(small_image, pad_width=int(w/2), mode='constant', constant_values=0)

    resized_image = cv2.resize(small_image, (28, 28))

    

    results.append(resized_image)

    

    plt.subplot(1, len(contours),i+1)

    plt.imshow(results[i], cmap=plt.cm.binary) 

    
results = np.array(results)

results = results.reshape(len(contours) ,results.shape[1] * results.shape[1])



print(results.shape)
output = pd.DataFrame(results,columns=[("pixel"+str(i)) for i in range(28*28)])



output.to_csv('nums.csv', index=False)

print("-- Your submission was successfully saved!")