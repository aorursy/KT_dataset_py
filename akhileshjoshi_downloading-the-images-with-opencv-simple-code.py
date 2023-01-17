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
'''
import cv2
import urllib.request 
import numpy as np

IMG_SIZE = 224 #set the image size

trainImagesFolder = "D:/Machine Learning Projects/Google Landmark Recognition Challenge/trainImages/" #use your local path

train_links=train_data['url']

print("Number of URLS" , len(train_links))

i=0

for link in train_links :
    if i%10000 == 0 :
        print("Iterations completed" , i)
    try:
        resp = urllib.request .urlopen(link)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)  # i am reading image in grayscale
        image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
        imgName = str(i)  + ".jpg"
        cv2.imwrite( trainImagesFolder + imgName ,image)
        i += 1
    except :
        imgName = str(i)  + ".jpg" #in case of bad HTTP I am saving the previous image (you can keep track of these false images with value of i)
        cv2.imwrite( trainImagesFolder + imgName ,image)
        i += 1
'''
