# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import scipy.misc as smp

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print('data({0[0]}, {0[1]})'.format(data.shape))

print(data.head())

print('-'*40)

print(test.head())
images = data.iloc[:,1:].values

images = images.astype(np.float)



#convert from [0, 255] to [0, 1]

images = np.multiply(images, 1/255)

print('data({0[0]}, {0[1]})'.format(images.shape))

print(images)
image_size = images.shape[1]

print('image size: {0}'.format(image_size))

image_width = image_height = np.ceil(math.sqrt(image_size)).astype(np.uint8)

print('image width: {0}\nimage height: {1}'.format(image_width, image_height))
# Plot average shapes of numbers



fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7), (ax8, ax9)) = plt.subplots(5, 2, figsize=(7, 10))

ax_list = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

avg_shapes_of_num = data.groupby(['label'], as_index=False).mean().drop(['label'], axis=1)



for i, ax in enumerate(ax_list):

    image_i = avg_shapes_of_num.iloc[i].reshape(image_width, image_height)

    ax.imshow(image_i, cmap=cm.hot)

    ax.axis('off')
# Prepare for training and test data sets

X_train = data.drop(['label'], axis=1)

y_train = data['label']

X_test = test
# Logistic Regression takes too much time



#logreg = LogisticRegression()

#logreg.fit(X_train, y_train)



#y_pred = logreg(X_test)

#print(logreg.score(X_train, y_train))
#svc = SVC()

#svc.fit(X_train, y_train)



#y_pred = svc.predict(X_test)



#print(svc.score(X_train, y_train))
random_forest = RandomForestClassifier(n_estimators = 100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)



print(random_forest.score(X_train, y_train))
#knn = KNeighborsClassifier(n_neighbors=10)

#knn.fit(X_train, y_train)

#y_pred = knn.predict(X_test)



#print(knn.score(X_train, y_train))
#gaussian = GaussianNB()

#gaussian.fit(X_train, y_train)

#y_pred = gaussian.predict(X_test)



#print(gaussian.score(X_train, y_train))
submission = pd.DataFrame({'ImageId': range(1, len(X_test)+1),'label': y_pred})

submission.to_csv('digit_recognizer.csv', index=False)