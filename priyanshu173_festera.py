# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 

import seaborn as sns

train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv')

test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv')

trainlabel = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv')

print(plt.style.available)

plt.style.use('ggplot')
print("train", train.shape)

print("test", test.shape)

print("trainLabel", trainlabel.shape)

train.head()

train.info()
train.describe()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, train_test_split



X, y = train, np.ravel(trainlabel)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
neig = np.arange(1, 25)

kfold = 10

train_accuracy = []

val_accuracy = []

bestKnn = None

bestAcc = 0.0

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(X_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(X_train, y_train))

    # test accuracy

    val_accuracy.append(np.mean(cross_val_score(knn, X, y, cv=kfold)))

    if np.mean(cross_val_score(knn, X, y, cv=kfold)) > bestAcc:

        bestAcc = np.mean(cross_val_score(knn, X, y, cv=10))

        bestKnn = knn

        

# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, val_accuracy, label = 'Validation Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('k value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.show()



print('Best Accuracy without feature scaling:', bestAcc)

print(bestKnn)