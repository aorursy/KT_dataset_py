import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

from skimage.color import rgb2gray

from skimage import feature

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.decomposition import PCA
df = np.load('/kaggle/input/eval-lab-4-f464/train.npy', allow_pickle=True)

test = np.load('/kaggle/input/eval-lab-4-f464/test.npy', allow_pickle=True)
df[0][1].shape #height, width, channels
h = df[0][1].shape[0]

w = df[0][1].shape[1]
df_gray = np.ndarray((df.shape[0], h, w))

test_gray = np.ndarray((test.shape[0], h, w))
df_gray
test_gray
for i, ii in enumerate(df):

    df_gray[i, :, :] = rgb2gray(ii[1])

for i, ii in enumerate(test):

    test_gray[i, :, :] = rgb2gray(ii[1])   
y = np.ndarray(df.shape[0])

y_pred = np.ndarray(test.shape[0])
le = LabelEncoder()

y = le.fit_transform(df[:, 0])
y
df_daisy = np.ndarray((df.shape[0], 5, 5, 200))

test_daisy = np.ndarray((test.shape[0], 5, 5, 200))

for i in range(df.shape[0]):

    df_daisy[i, :, :, :] = feature.daisy(df_gray[i])

for i in range(test.shape[0]):

    test_daisy[i, :, :, :] = feature.daisy(test_gray[i])
X = df_daisy.reshape(df.shape[0], 5 * 5 * 200)

test2 = test_daisy.reshape(test.shape[0], 5 * 5 * 200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
pca = PCA(n_components = 200, svd_solver = 'randomized', whiten = True).fit(X)

X_pca = pca.transform(X)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)

test2_pca = pca.transform(test2)
from time import time

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5] }

clf = GridSearchCV(SVC(kernel='rbf', gamma = 'scale', class_weight='balanced'),

                   param_grid,)

clf = clf.fit(X_pca, y)

print(clf.best_estimator_)
pred = le.inverse_transform(clf.predict(test2_pca))
Imageids = test[:, 0]

df_submit = pd.DataFrame()

df_submit['ImageId'] = Imageids

df_submit['Celebrity'] = pred

df_submit.head()
df_submit.to_csv('final.csv', index = False)