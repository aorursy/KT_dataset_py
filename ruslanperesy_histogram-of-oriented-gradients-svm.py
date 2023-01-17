# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.externals import joblib

from skimage.feature import hog

from sklearn.svm import LinearSVC



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



train_set = pd.read_csv("../input/train.csv")

#train_set.shape

#train_set.head()



features = np.array(train_set.iloc[:,1:], 'int16')

#features.shape

labels = np.array(train_set.iloc[:,0], 'int')

#labels.shape



list_hog_fd = []

for feature in features:

    fd = hog(feature.reshape((28,28)), orientations=9, pixels_per_cell=(14,14), cells_per_block=(1, 1), visualize=False)

    list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')



clf = LinearSVC()

clf.fit(hog_features, labels)



joblib.dump(clf, "digits_cls.pkl", compress=3)
# Testing the Classifier

# Import the modules

# WARNING you need to run the first cell before to obtain a classifier

from sklearn.externals import joblib

from skimage.feature import hog

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



# Load the classifier

clf = joblib.load("digits_cls.pkl")



# Load the test set

test_set = pd.read_csv("../input/test.csv")

digits = np.array(test_set, 'int16')



results = []

for digit in digits:

    dg_hog_fd = hog(digit.reshape((28,28)), orientations = 9, pixels_per_cell=(14,14), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=False)

    nbr = clf.predict(np.array([dg_hog_fd], 'float64'))

    results.append(nbr)

    

i = 8

img = test_set.iloc[i].as_matrix()

img = img.reshape((28,28))

plt.imshow(img, cmap='gray')

plt.title(results[i])
