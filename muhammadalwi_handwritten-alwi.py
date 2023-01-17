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
        
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('/kaggle/input/handwritten/handwritting.jpg')
image = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)

#membuat image baru dengan intensitas nol


h,w = image.shape
gr = np.zeros([h,w], dtype=np.uint8)

#proses binerisasi

t = 65
for y in range(0, h):
    for x in range(0,w):
        v = int(image[y,x])
        if v < t :
            img[y,x] = 0
        elif v >= t:
            img[y,x] = 255
        
plt.imshow(img,'gray')
plt.show()

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session