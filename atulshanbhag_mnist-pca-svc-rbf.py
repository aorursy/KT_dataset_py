# imports

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.svm import SVC
# data locations

train_loc = "../input/train.csv"

test_loc = "../input/test.csv"
train = pd.read_csv(train_loc)

X = train.drop("label", axis=1).as_matrix()

train_size = X.shape[0]

y = train["label"].values
test = pd.read_csv(test_loc).as_matrix()

test_size = test.shape[0]
img_idx = np.random.randint(test_size, size=6)

fig = plt.figure()

for i in range(6):

    instance = test[img_idx[i]]

    assert(len(instance) == 784) # making sure we have correct image instances

    img = instance.reshape(28, 28)

    plt.subplot(230 + i + 1) # creating a 2x3 image-grid for the images

    plt.title(img_idx[i] + 1) # ImageId

    plt.axis("off")

    plt.imshow(img, cmap=cm.gray)
# X_train, y_train are the training sets and labels used in our model

# X_test, y_test are the validation sets for our model

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, 

                                                   random_state=50)
# n_components=33 was determined by running a brute-force

# over various values, and comparing the best scores

# very time consuming

pca = PCA(n_components=33, whiten=True)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

test = pca.transform(test)
svc = SVC(C=10, kernel="rbf", # the parameters C and gamma were obtained 

          gamma=0.05,        # by using cross validation and grid search method

          class_weight="balanced", 

          random_state=50)
svc.fit(X_train, y_train)

score = svc.score(X_test, y_test)

print("Score {0:.4f}".format(score))
labels = svc.predict(test)
res = pd.DataFrame(columns=["ImageId", "Label"])

res["ImageId"] = pd.Series(1 + np.arange(test_size))

res["Label"] = pd.Series(labels)
res.iloc[img_idx]
res.to_csv("submission.csv", index=False)