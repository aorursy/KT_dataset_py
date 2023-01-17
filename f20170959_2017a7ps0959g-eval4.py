import pandas as pd

import numpy as np

from skimage.feature import daisy

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

%matplotlib inline
data = np.load(file = 'train.npy', allow_pickle = True)
from skimage import exposure

train, labels = [], []

for i in data:

    train.append(daisy(exposure.equalize_hist(i[1].mean(axis=2))).flatten())

    labels.append(i[0])

train = np.array(train)
train.shape
test = np.load(file = 'test.npy', allow_pickle = True)
test_f, id = [], []

for i in test:

    test_f.append(daisy(exposure.equalize_hist(i[1].mean(axis=2))).flatten())

    id.append(i[0])
pca = PCA(n_components=80, whiten=True)
pca.fit(train)

X_f_pca = pca.transform(np.array(train))

test_f_pca = pca.transform(np.array(test_f))
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

clf = clf.fit(X_f_pca, labels)
y_f_pred = clf.predict(test_f_pca)
sub_5 = pd.DataFrame()

sub_5['ImageId'] = id

sub_5['Celebrity'] = y_f_pred

sub_5.to_csv('submission_5.csv', sep = ',', index = None)