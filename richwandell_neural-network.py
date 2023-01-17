import numpy as np 

import pandas as pd



train = pd.read_csv('../input/train.csv')

# create target 

y = np.array(train['label'], dtype=str)

# create input array 

X = np.array(train.drop('label', axis=1), dtype=float)

# normalize

X = X / 255

# select 75% for training, we will use the other 25% to test

train_size = int(X.shape[0] * .75)
import matplotlib.pyplot as plt



def gimg(i):

    # images must be 28x28x3

    return np.reshape(

        # greyscale images using the same value for R G B

        np.column_stack(

            (X[i], X[i], X[i])

        ),

        (28, 28, 3)

    )



# create the top 5

img = gimg(0)

for i in range(1, 5):

    img = np.column_stack((img, gimg(i)))

    

# create the bottom 5

img1 = gimg(6)

for i in range(7, 11):

    img1 = np.column_stack((img1, gimg(i)))



# add bottom to the top and swap sign 

img = 1 - np.row_stack((img, img1))



plt.imshow(img)
from sklearn.decomposition import PCA



pca = PCA(n_components=50)

training_data = pca.fit_transform(X[:train_size], y[:train_size])
from sklearn.neural_network import MLPClassifier



clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='relu', max_iter=3000,

                    hidden_layer_sizes=(30,), random_state=1)

clf.fit(training_data, y[:train_size].ravel())
from sklearn import metrics



predicted = clf.predict(pca.transform(X[train_size:]))

actual = y[train_size:]

print(metrics.classification_report(actual, predicted))

print(metrics.confusion_matrix(actual, predicted))