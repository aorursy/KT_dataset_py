# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from matplotlib import pyplot as plt

from PIL import Image





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img = cv2.imread('/kaggle/input/sudoku/image.png',0)

img = cv2.medianBlur(img,5)



#ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\cv2.THRESH_BINARY,11,2)

th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\

            cv2.THRESH_BINARY,11,2)



titles = ['Original Image', 'Global Thresholding (v = 127)',

            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']

images = [img,th3]



for i in range(2):

    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')

    plt.title(titles[i])

    plt.xticks([]),plt.yticks([])

plt.show()


def canny(path):

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.GaussianBlur(img, (7,7), 3)

    newimg = cv2.Canny(img, 80, 120)

    return newimg

    



plt.imsave("canny.jpg", canny("/kaggle/input/sudoku/image.png"))

    