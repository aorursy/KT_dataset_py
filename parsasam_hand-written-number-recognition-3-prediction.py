# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.models import load_model



model = load_model('/kaggle/input/hand-written-number-recognition-2-mnist-model/digit_recognition_model/')

print('-- Model loaded')
num = pd.read_csv("/kaggle/input/hand-written-number-recognition-1-image-to-digit/nums.csv")

num.head()
num = np.array(num)

num = num.astype('float32') / 255



print("-- Data Prepared")
result = model.predict(num)

result = [np.argmax(i) for i in result]



print("-- Predicted")
import cv2

import matplotlib.pyplot as plt



image = cv2.imread('/kaggle/input/hand-written-digits/unnamed.jpg')



plt.imshow(image) 

plt.show() 
print('Here You Go ! :')

for n in result:

    print(n, end=" ")