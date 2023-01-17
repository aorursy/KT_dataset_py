%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier
# read in the data



train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

# prepare data



labels = train["label"]

train = train.drop("label",1)

X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=0)

# show samples



for digit_num in range(0,64):

    plt.subplot(8,8,digit_num+1)

    grid_data = train.iloc[digit_num].as_matrix().reshape(28,28)

    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")

    plt.xticks([])

    plt.yticks([])
np.random.seed(12345)

classifier = RandomForestClassifier(n_estimators=100)

classifier = classifier.fit(X_train, y_train)



#results = classifier.predict(X_test)

classifier.score(X_test, y_test)
knn = KNeighborsClassifier()

knn = knn.fit(X_train, y_train)



knn.score(X_test, y_test)