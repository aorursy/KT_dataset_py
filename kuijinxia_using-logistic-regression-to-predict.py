import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

%matplotlib inline

np.random.seed(2)

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Activation,Input,MaxPool2D,Conv2D,Flatten,Dropout,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model

sns.set(style='white',context='notebook',palette='deep')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
Y_train = train["label"]

X_train = train.drop(["label"], axis=1 )

g = sns.countplot(Y_train)

Y_train.value_counts()
X_train /= 255.0
test /= 255.0

train_image = X_train.values.reshape(-1,784)
train_labels = Y_train
test_image = test.values.reshape(-1,784)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_image, train_labels)
# print(clf.score(test_image,test_labels))
pre = clf.predict(test_image)
pre_pd = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pd.Series(pre,name = "Label")])
pre_pd.to_csv("my_submission.csv",index=False)


