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

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))


import matplotlib.pyplot as plt


data=pd.read_csv('../input/train.csv')
data.head()
data.shape
image1=data['Image'].loc[0]
print(image1)
import imageio

path_join = os.path.join('../input/train', image1)
image = imageio.imread(path_join)
imgplot = plt.imshow(image)
plt.title("first image")
Image_column=np.array(data.Image)
print(Image_column)
Image_column.size

          



    







X=data.iloc[:,:-1].values #all the image
y = data.iloc[:, 1].values #whale classification
print(X.shape)
print(y.shape)

X_train,y_train,X_test,y_test=train_test_split(X, y, test_size=0.30, random_state=42)
print(X_train.shape)
print(y_train.shape)
X_train[77]
import matplotlib.pyplot as plt
%matplotlib inline 
image_index = 77 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
img=X_train[image_index]
pixels = X_train.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.figure()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(data)
plt.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="Paired")
plt.colorbar()

