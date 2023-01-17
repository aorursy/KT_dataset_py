# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

labeled_images = pd.read_csv('../input/train.csv')

images = labeled_images.iloc[0:10000,1:]

labels = labeled_images.iloc[0:10000,:1]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.7, random_state=0)
import matplotlib.pyplot as plt, matplotlib.image as mpimg

i=1

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
test_images[test_images>0]=1

train_images[train_images>0]=1



img=train_images.iloc[i].as_matrix().reshape((28,28))

plt.imshow(img,cmap='binary')

plt.title(train_labels.iloc[i])
import matplotlib.pyplot as plt, matplotlib.image as mpimg

i=1

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
test_images[test_images>0]=1

train_images[train_images>0]=1



img=train_images.iloc[i].as_matrix().reshape((28,28))

plt.imshow(img,cmap='binary')

plt.title(train_labels.iloc[i])
from sklearn.decomposition import PCA

n_components = 150;

pca = PCA(svd_solver='randomized', n_components=n_components)

pca.fit(train_images)

train_images_pca = pca.transform(train_images)

test_images_pca = pca.transform(test_images)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [1e3, 5e3, 5e4],

              'gamma': [0.0005, 0.005, 0.01, 0.1],

              'kernel': ['rbf', 'linear'], 

              'class_weight': ['balanced']}

svr = SVC()

clf = GridSearchCV(svr, param_grid)

c = train_labels.shape

train_labels = train_labels.values.reshape(c,)

#clf = SVC(kernel="linear",C=0.05)

clf = clf.fit(train_images_pca, train_labels[:,0])

pred = clf.predict(test_images_pca)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_labels, pred)

print ("accuracy:", accuracy)
print(clf.cv_results_.get('params'))
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),

                         "Label": pred})

submissions.to_csv("DR.csv", index=False, header=True)
print(clf.best_estimator_)