# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from scipy.ndimage import convolve

from sklearn import linear_model, datasets, metrics, svm, decomposition

from sklearn.neural_network import BernoulliRBM

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from PIL import Image
train = pd.read_csv("../input/train.csv")
train_images = (train.ix[:,1:].values).astype('float32')

train_labels = train.ix[:,0].values.astype('int32')

train_inputs = (train.ix[:,1:].values).astype('float32')

train_inputs = np.multiply(train_inputs, 1.0 / 255.0)



X_train = train_inputs[:10000]

Y_train = train_labels[:10000]



X_test = train_inputs[10000:15000]

Y_test = train_labels[10000:15000]
im = Image.open("../input/cat2.png").convert('L')

cat_img = np.array(im.getdata())/225.



plt.imshow(cat_img.reshape((28, 28)),cmap=plt.get_cmap('gray'),

               interpolation='nearest')

plt.suptitle('The Cat', fontsize=16)

plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)



plt.show()
cat_multi_img = np.tile(cat_img, (1000, 1))

X_train_cat = np.concatenate((cat_multi_img, X_train), axis=0)

X_train_cat = np.random.permutation(X_train_cat)
train_images = X_train_cat.reshape(X_train_cat.shape[0],  28, 28)



plt.figure(figsize=(8, 8))

for i in range(100):

    plt.subplot(10, 10, i + 1)

    plt.imshow(train_images[i], cmap=plt.get_cmap('gray'),

               interpolation='nearest')

    plt.xticks(())

    plt.yticks(())



   

plt.suptitle('100 components extracted by RBM', fontsize=16)

plt.subplots_adjust(0.08, 0.2, 0.92, 0.85, 0.08, 0.23)



plt.show()