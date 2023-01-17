import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.cm as cm

%matplotlib inline
data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
data.head()
numb1=data.loc[0]

y_numb1 = numb1[0]

x_numb1 = numb1[1:]

x_numb1 = x_numb1.values.reshape(28,28)
plt.matshow(x_numb1, cmap=plt.cm.gray)
labels = data['label']

data = data.drop('label',axis=1)

data.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=50)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, n_jobs=5)

clf.fit(x_train, y_train)

res=clf.predict(x_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,res)
from sklearn.metrics import confusion_matrix

conf = confusion_matrix(y_test, res)

conf
plt.matshow(conf, cmap=plt.cm.gray)
#It looks some are a little bit gray.

row_sums = conf.sum(axis =1, keepdims=True)

norm_conf = conf/row_sums

np.fill_diagonal(norm_conf,0)

plt.matshow(norm_conf,cmap=plt.cm.gray)

plt.show()
# this function draw images

def plot_digits(instances, images_per_row=10, **options):

    size = 28

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in instances]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = matplotlib.cm.binary, **options)

    plt.axis("off")
a,b=2,3

x_aa = x_test[(y_test == a) & (res ==a)].values

x_ab = x_test[(y_test == a) & (res ==b)].values

x_ba = x_test[(y_test == b) & (res ==a)].values

x_bb = x_test[(y_test == b) & (res ==b)].values



plt.subplot(221)

plot_digits(x_aa[:25], 5)

plt.subplot(222)

plot_digits(x_ab[:25], 5)

plt.subplot(223)

plot_digits(x_ba[:25], 5)

plt.subplot(224)

plot_digits(x_bb[:25], 5)

#submit

clf = RandomForestClassifier(n_estimators=150, n_jobs=10)

clf.fit(data, labels)

res = clf.predict(test)
import csv

exp_res=np.exp(res)

submission=pd.DataFrame({'ImageId':test.index+1, 'Label':res})

submission.to_csv('RF.csv', sep=',', index=False)