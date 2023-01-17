# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        print(os.listdir("../input"))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



train = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')

test = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/test.csv')

icml_face_data = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
print(train)
print(test)
#train.head()

train_x = train.pixels

train_y = train.emotion

print(train)
%matplotlib inline

#plt.imshow(train.pixels[10])

print(train.pixels[10])
#for image_pixels in train.iloc[1:,1]: #column 2 has the pixels. Row 1 is column name.

#    image_string = image_pixels.split(' ') #pixels are separated by spaces.

#    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)

   
image_string = train_x[1000].split(' ') 

image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)

plt.imshow(image_data)