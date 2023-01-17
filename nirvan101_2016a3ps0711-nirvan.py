# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import cv2
df = np.load('/kaggle/input/eval-lab-4-f464/train.npy',allow_pickle=True)

test = np.load('/kaggle/input/eval-lab-4-f464/test.npy',allow_pickle=True)
#check sample submission

import os

path = '/kaggle/input/eval-lab-4-f464/sample_submission.csv'

sample = pd.read_csv(path)

sample.head()
names = []

images = []

for row in df:

    names.append(row[0])

    images.append(row[1])
images[0].shape
flat_images = []

for img in images:

    flat_images.append(img.reshape((50*50*3,)))

print(flat_images[0].shape)
test_flat_images = []

for row in test:

    test_flat_images.append(row[1].reshape((50*50*3,)))

print(test_flat_images[0].shape)
images_gray=[]

for img in images:

    images_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    

images_gray_test=[]

for img in test:

    images_gray_test.append(cv2.cvtColor(img[1], cv2.COLOR_BGR2GRAY))    

    

print(images_gray[0].shape)

print(images_gray_test[0].shape)

    
#from sklearn.neighbors import KNeighborsClassifier

#knn_clf = KNeighborsClassifier(n_neighbors=3)

#knn_clf.fit(images_gray, names)
images_gray = np.array(images_gray)

images_gray = images_gray.reshape((2275,2500))
images_gray_test = np.array(images_gray_test)

images_gray_test = images_gray_test.reshape((976,2500))
flat_images = np.array(flat_images)

test_flat_images = np.array(test_flat_images)
flat_images.shape
from sklearn.decomposition import PCA as RandomizedPCA



X_train = flat_images #images_gray

# print "Extracting the top %d eigenfaces" % n_components

pca = RandomizedPCA(n_components=80, whiten=True).fit(X_train)

X_test = test_flat_images #images_gray_test

# eigenfaces = pca.components_.T.reshape((n_components, 64, 64))



# project the input data on the eigenfaces orthonormal basis

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



param_grid = {

 'C': [1, 5, 10, 50, 100],

 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 'scale']

}



clf = GridSearchCV(SVC(kernel='rbf'), param_grid)



#clf = SVC(C=4, gamma=0.004, random_state=42 ,kernel='rbf')

clf = clf.fit(X_train_pca, names)
'''

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

label_encoder = label_encoder.fit(names)

label_encoded_names = label_encoder.transform(names) 

'''
#label_encoded_names
import xgboost



#model = xgboost.XGBClassifier()

#model.fit(X_train_pca, label_encoded_names)



#D_train = xgb.DMatrix(X_train_pca, label=names_xgb)

#D_test = xgb.DMatrix(X_test_pca)



'''

param = {

    'eta': 0.3, 

    'max_depth': 3,  

    'objective': 'multi:softprob',  

    'num_class': 3} 

'''





#clf = GridSearchCV(SVC(kernel='rbf'), param_grid)

#clf = clf.fit(X_train_pca, names)

#steps = 20  # The number of training iterations



#xgb_pred = model.predict(X_test_pca)



'''

model = xgboost.XGBClassifier()

parameters = {'nthread':[6], #when use hyperthread, xgboost may become slower

              'objective':['binary:logistic'],

              'learning_rate': [0.05], #so called `eta` value

              'max_depth': [6,20,40],

              'min_child_weight': [11],

              'silent': [1],

              'subsample': [0.8],

              'colsample_bytree': [0.7],

              'n_estimators': [1000], #number of trees, change it to 1000 for better results

              'missing':[-999],

              'seed': [1337,130]}





clf = GridSearchCV(model, parameters, n_jobs=5,  

                   scoring='roc_auc',

                   verbose=2, refit=True)

clf = clf.fit(X_train_pca, label_encoded_names)

'''
#xgb_pred = label_encoder.inverse_transform(xgb_pred)
print("Best estimator found by grid search:")

print(clf.best_estimator_)
y_pred = clf.predict(X_test_pca)
y_pred
len(y_pred)
out = pd.DataFrame(data={'ImageId':np.arange(0,976),'Celebrity':y_pred})

out.to_csv('submission.csv',index=False)
out.head()