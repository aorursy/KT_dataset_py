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
import pandas as pd

mnist_ds = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X = mnist_ds.drop(columns=["label"])

y = mnist_ds["label"]



#perform data augumentation - shift all images by one pixel

X_2 = X.shift(periods=1, axis='columns', fill_value=0)

X = X.append(X_2)

y = y.append(y)





import matplotlib as mpl

import matplotlib.pyplot as plt



X_np = X.to_numpy()

some_digit = X_np[3]

some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")

plt.axis("off")

plt.show()



def plot_digit(data):

    image = data.reshape(28,28)

    plt.imshow(image, cmap=mpl.cm.binary, interpolation="nearest")

    plt.axis("off")



# y is stored as a string -- convert to integer

y = y.astype(np.uint8)







from sklearn.preprocessing import StandardScaler

X_np = StandardScaler().fit_transform(X_np)



from sklearn import decomposition

pca = decomposition.PCA(n_components=100)

pca.fit(X_np)

X_np = pca.transform(X_np) 



# # split the set between train and validation

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_np, y, test_size=0.2, random_state=42, shuffle=True)



# from sklearn.model_selection import GridSearchCV

# classifier = KNeighborsClassifier()

# param_grid = {'n_neighbors': [5, 10], 'weights':['distance'], 'metric':['euclidean']}

# gs = GridSearchCV(classifier, param_grid, verbose=1, cv=3, n_jobs=-1)

# gs.fit(X_train, y_train)

# print(gs.best_params_)

# y_pred = gs.predict(X_test)



from sklearn.neighbors import KNeighborsClassifier

#use weights parameters to increase performance

classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)





from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
