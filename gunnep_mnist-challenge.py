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
import csv
#28 x 28 images, in total 42000 for training
import pandas as pd
csv_file = '/kaggle/input/digit-recognizer/train.csv'
df = pd.read_csv(csv_file)
y_train = df.label
x_train = df.drop(['label'],axis=1)   
arr_y_train = y_train.to_numpy()
arr_x_train = x_train.to_numpy()

#reshape the images into vectors
arr_x_train = arr_x_train.reshape(42000, 28, 28)
#example how to display an image from a vector
from matplotlib import pyplot as plt
plt.imshow(arr_x_train[2], interpolation='nearest')
plt.show()
#ready to develop a NN

