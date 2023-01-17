# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visulization 
from sklearn import svm  # Support vector machine
from sklearn.model_selection import train_test_split  # split the train data

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#loading data files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.shape
#exploring data files
train_image = train.iloc[0:5000,1:]
train_label = train.iloc[0:5000,:1]
train.head()
#Since the image is currently one-dimension, we load it into a numpy array and reshape it so that it is two-dimensional (28x28 pixels)
i=0
train_img = train_image.iloc[i].as_matrix()
train_img = train_img.reshape((28,28))
plt.imshow(train_img, cmap = 'gray')
train_label.iloc[i]
train_img1.shape
#Split the training data into train & validation set
train_images, val_images , train_labels, val_labels = train_test_split(train_image, train_label,train_size = 0.8, random_state =0)
#trying SVM 
clf = svm.SVC()
#fitting the modle
clf.fit(train_images,train_labels.values.ravel())
#checking the accuracy for validation set
clf.score(val_images, val_labels)
train_image/=255
test_SVM=test/255
train_images/=255
val_images/=255
clf.fit(train_images,train_labels.values.ravel())
clf.score(val_images,val_labels)
results_SVM=clf.predict(test_SVM[0:])
#convert the result into desierd file(.csv)
df = pd.DataFrame(results_SVM)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('../results_SVM.csv', header=True)
df.head()
#Exploring orignal data files and converting them itno array
x_train = np.array(train.iloc[:,1:])
y_train = np.array(train.iloc[:,:1])
x_test = np.array(test)
n_features_train = x_train.shape[1]        #[1]represent col
n_samples_train = x_train.shape[0]        #[0]represent row
n_features_test = x_test.shape[1]
n_samples_test = x_test.shape[0]
print(n_features_train, n_samples_train, n_features_test, n_samples_test)
print(x_train.shape, y_train.shape, x_test.shape)
# show the image
def show_img(X):
    plt.figure(figsize=(8,7))
    n_samples = X.shape[0]
    X = X.reshape(n_samples, 28, 28)
    for i in range(20):
        plt.subplot(5, 4, i+1)
        plt.imshow(X[i])
    plt.show()
show_img(x_train)
show_img(x_test)
