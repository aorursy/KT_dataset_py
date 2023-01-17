import pandas as pd

import numpy as np

from sklearn import svm, decomposition, preprocessing



train_df = pd.read_csv('../input/train.csv', header=0)



train_x = train_df.values[:, 1:]

train_y = train_df.ix[:,0]



print("size of training set: ", len(train_y))
import matplotlib.pyplot as plt

from skimage.transform import rotate



ARTIFICIAL_DATA_SIZE = 2500

n = len(train_y)



rnd_indexes = np.random.randint(0, n, size=ARTIFICIAL_DATA_SIZE)

train_x = np.vstack((train_x, np.zeros((ARTIFICIAL_DATA_SIZE, train_x.shape[1]))))



k = n

for ind in rnd_indexes:

    img = train_x[ind, :].copy().reshape(28,28)

    angle = np.random.randint(-20, 20)

    img_rot = rotate(img, angle)

    train_x[k,:] = img_rot.reshape(784,)

    train_y[k] = train_y[ind]

    k = k + 1

print("size of training set after random rotations: ", len(train_y))

plt.title("last of added training examples, label: " + str(train_y[k-1]))

img1 = plt.imshow(train_x[k-1,:].copy().reshape(28,28), cmap=plt.cm.gray_r, interpolation="nearest")
COMPONENTS_RATIO= 0.8



train_x = preprocessing.scale(train_x)



pca = decomposition.PCA(n_components = COMPONENTS_RATIO, whiten = False)

train_x = pca.fit_transform(train_x)
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_x,train_y,random_state = 42,

                                                    test_size = 0.1)



classifier = svm.SVC(C=2.0)

classifier.fit(X_train, Y_train)

score = classifier.score(X_test, Y_test)

print("SVM score: ", score)

                                     
test_df  = pd.read_csv("../input/test.csv")

test_x = test_df.values

test_x = preprocessing.scale(test_x)

test_x = pca.transform(test_x)

test_pred = classifier.predict(test_x)
pd.DataFrame({"ImageId": range(1,len(test_pred)+1), "Label": test_pred}).to_csv('out.csv', index=False, header=True)    