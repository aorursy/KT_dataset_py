import pandas as pd

import sklearn as sk

from sklearn import svm, decomposition, preprocessing

import numpy as np
train = pd.read_csv("../input/train.csv")

#train = train.as_matrix()

X = train.values[:,1:]

Y = train.ix[:,0]

print(len(Y))
import matplotlib.pyplot as plt

from skimage.transform import rotate





n = len(Y)

print(n)



random_indexes = np.random.randint(0, high = n, size = 10000 )

X = np.vstack((X, np.zeros((10000, X.shape[1]))))

Y = np.concatenate([Y,np.zeros((10000))])

print(Y.shape)

for ind in random_indexes:  

    img = X[ind, :].reshape(28,28)

    angle = np.random.randint(-30, 30)

    img_rot = rotate(img, angle)

    X[n,:] = img_rot.reshape(1,784)

    Y[n] = Y[ind]

    n = n + 1

print("size of training set after random rotations: ", X.shape)

print(len(Y))

plt.title("last of added training examples, label: " + str(Y[-1]))

img1 = plt.imshow(X[n-1,:].reshape(28,28), cmap=plt.cm.gray_r, interpolation="nearest")





X = preprocessing.scale(X)
COMPONENTS_NUM = 300



pca = decomposition.PCA(n_components = 0.8, whiten = True)

X = pca.fit_transform(X)
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state = 7,

                                                    test_size = 0.3)



classifier = svm.SVC()

classifier.fit(X_train, Y_train)

score = classifier.score(X_test, Y_test)

print('Score \n', score)

                                                    

                                                    

                                                   
test  = pd.read_csv("../input/test.csv")

test = test.values

test = preprocessing.scale(test)

#pca = decomposition.PCA(components_num)

X_test_red = pca.transform(test)

Y_pred = classifier.predict(X_test_red)
pd.DataFrame({"ImageId": range(1,len(Y_pred)+1), "Label": Y_pred}).to_csv('out.csv', index=False, header=True)    