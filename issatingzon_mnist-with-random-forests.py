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
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.grid_search import GridSearchCV

from sklearn import preprocessing

import matplotlib.pyplot as plt
seed = 42

train = '../input/train.csv'

dataframe = pd.read_csv(train, header=0) 

X = dataframe.iloc[:, 1:]

y = dataframe.iloc[:, 0]
i = 8 #8th row in the dataset

img = X.iloc[i].as_matrix()

img = img.reshape((28, 28))

plt.imshow(img, cmap='gray_r')

plt.title('Class label: ' + str(dataframe.iloc[i, 0]))

plt.show()
def split(X, y):

    train_size = 0.8

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed)

    return X_train, X_test, y_train, y_test
def eval(X_train, y_train):

    classifiers = dict() 

    classifiers['Gaussian Naive Bayes'] = GaussianNB()

    classifiers['Decision Tree Classifier'] = DecisionTreeClassifier(random_state=seed)

    classifiers['Random Forests'] = RandomForestClassifier(max_depth=2, random_state=0)



    for clf_name, clf in classifiers.items():

        clf.fit(X_train, y_train)

        score = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy').mean()

        print(clf_name, score)
#X_train, X_test, y_train, y_test = split(X,y)

#eval(X_train, y_train)
binarizer = preprocessing.Binarizer()

X_binarized = binarizer.transform(X)

X_binarized = pd.DataFrame(X_binarized)



i = 8

img = X_binarized.iloc[i].as_matrix()

img = img.reshape((28, 28))

plt.imshow(img, cmap='binary')

plt.title('Class label: ' + str(y[i]))

plt.show()
X_train, X_test, y_train, y_test = split(X_binarized, y)

#eval(X_train, y_train)
clf = RandomForestClassifier(n_jobs=1, n_estimators=500, max_features='auto',random_state=seed)

clf.fit(X_binarized, y)

score = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy').mean()

print(score)
test_data = pd.read_csv('../input/test.csv')

test_binarized = binarizer.transform(test_data)

results = clf.predict(test_binarized[:])
df = pd.DataFrame(results)

df.index += 1 #indexing starts from 1

df.index.names = ['ImageId']

df.columns = ['Label']

df.to_csv('results.csv', header=True)