from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.datasets import fetch_lfw_people

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, f1_score

from sklearn.decomposition import PCA

from sklearn.svm import SVC

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2
np_train = np.load('./train.npy', allow_pickle=True)

np_test  = np.load('./test.npy', allow_pickle=True)



x_train = np_train[:,1]

y_train = np_train[:,0]

x_test  = np_test[:,1]

ImageIds= np_test[:,0]



X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
from skimage.feature import hog

from skimage.io import imread

from skimage.transform import rescale



from sklearn.base import BaseEstimator, TransformerMixin



class RGB2GrayTransformer(BaseEstimator, TransformerMixin):

    """

    Convert an array of RGB images to grayscale

    """



    def __init__(self):

        pass



    def fit(self, X, y=None):

        """returns itself"""

        return self



    def transform(self, X, y=None):

        """perform the transformation and return an array"""

        return np.array([skimage.color.rgb2gray(img) for img in X])





class HogTransformer(BaseEstimator, TransformerMixin):

    """

    Expects an array of 2d arrays (1 channel images)

    Calculates hog features for each img

    """



    def __init__(self, y=None, orientations=9,

                 pixels_per_cell=(8, 8),

                 cells_per_block=(3, 3), block_norm='L2-Hys'):

        self.y = y

        self.orientations = orientations

        self.pixels_per_cell = pixels_per_cell

        self.cells_per_block = cells_per_block

        self.block_norm = block_norm



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):



        def local_hog(X):

            return hog(X,

                       orientations=self.orientations,

                       pixels_per_cell=self.pixels_per_cell,

                       cells_per_block=self.cells_per_block,

                       block_norm=self.block_norm)



        try: # parallel

            return np.array([local_hog(img) for img in X])

        except:

            return np.array([local_hog(img) for img in X])





from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import StandardScaler

import skimage



# create an instance of each transformer

grayify = RGB2GrayTransformer()

hogify = HogTransformer(

    pixels_per_cell=(8, 8),

    cells_per_block=(2,2),

    orientations=9,

    block_norm='L2-Hys'

)

scalify = StandardScaler()



# call fit_transform on each transform converting X_train step by step

X_train_gray = grayify.fit_transform(X_train)

X_train_hog = hogify.fit_transform(X_train_gray)

X_train_prepared = scalify.fit_transform(X_train_hog)



print(X_train_prepared.shape)



from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(X_train_prepared, Y_train)
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=2000)

clf.fit(X_train_prepared, Y_train)
from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=42, tol=1e-5, )

clf.fit(X_train_prepared, Y_train) 
clf = KNeighborsClassifier(n_neighbors=2)

clf.fit(X_train_prepared, Y_train)
X_test_gray = grayify.transform(x_test)

X_test_hog = hogify.transform(X_test_gray)

X_test_prepared = scalify.transform(X_test_hog)

y_pred = clf.predict(X_test_prepared)



d = {'ImageId':ImageIds, 'Celebrity':y_pred}

finalAnswer = pd.DataFrame(data=d, index=None)

finalAnswer.to_csv('sub101.csv',index=None)
X_val_gray = grayify.fit_transform(X_val)

X_val_hog = hogify.fit_transform(X_val_gray)

X_val_prepared = scalify.fit_transform(X_val_hog)

y_pred = clf.predict(X_val_prepared)

# print(len(np.unique(y_pred,return_counts=True)[0]))

target_names = np.unique(y_train,return_counts=True)[0]

classes = np.unique(y_train,return_counts=True)[0]

print(classification_report(Y_val, y_pred, target_names=target_names))

print(confusion_matrix(Y_val, y_pred, labels=classes))

print(f1_score(Y_val, y_pred, average='micro'))
y_pred = clf.predict(X_test_prepared)

for i in range(0,3):

    plt.figure(figsize=(10,5))

    for j in range(0,5):

        plt.subplot(1,5,j+1)

        plt.title(y_pred[5*i+j],loc='center')

        plt.imshow(x_test[5*i+j])

    plt.show()
x = np.unique(np_train[:,0],return_counts=True)

labels = x[0]

counts = x[1]

plt.figure(figsize=[25,5])

plt.bar(labels, counts)

plt.yticks(np.arange(0,300,50))

plt.grid(axis='y')

plt.show()