# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
test.head()
train_x = train.iloc[:, 1:].values.astype('float32')

train_y = train.iloc[:, 0].values.astype('int32')

test_x = test.values.astype('float32')
train_imgs = train_x.reshape((-1, 1, 28, 28)) / 255.

test_imgs = test_x.reshape((-1, 1, 28, 28)) / 255.
train_imgs = train_imgs.reshape((len(train_imgs), 784))

test_imgs = test_imgs.reshape((len(test_imgs), 784))
from sklearn.svm import SVC

clf = SVC(gamma=0.001, C=100.)

clf.fit(train_imgs, train_y)
test_predict = clf.predict(test_imgs)
test_predict
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
sample_submission.head()
sub = sample_submission

sub['Label'] = list(map(int, test_predict))

sub.to_csv("submission.csv", index=False)