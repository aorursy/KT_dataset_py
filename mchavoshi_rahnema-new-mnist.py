# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
print(train_df.shape)

train_df.head()
test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

print(test_df.shape)

test_df.head()
test_submit_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

test_submit_df.head()
df = pd.concat([train_df, test_df])

df.shape
X = df[df.columns[1:]]

y = df[df.columns[0]]
idx = 12

first_image = np.array(X.iloc[idx], dtype='float')

pixels = first_image.reshape((28, 28))

print(y.iloc[idx])

plt.imshow(pixels, cmap='gray')
df.describe()
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
dimen_reduction = PCA(n_components=60)

X_train_reduced = dimen_reduction.fit_transform(X_train)

X_test_reduced = dimen_reduction.transform(X_test)
dimen_reduction.explained_variance_ratio_
lr = LogisticRegression()

lr.fit(X_train_reduced, y_train)
accuracy_score(y_train, lr.predict(X_train_reduced))
accuracy_score(y_test, lr.predict(X_test_reduced))
clf_nb = GaussianNB()

clf_nb.fit(X_train, y_train)
accuracy_score(y_test, clf_nb.predict(X_test))
# clf_svm = SVC()

# clf_svm.fit(X_train, y_train)
accuracy_score(y_test, clf_svm.predict(X_test))
tree = DecisionTreeClassifier(max_depth=15)

tree.fit(X_train, y_train)
accuracy_score(y_test, tree.predict(X_test))
# clf = GridSearchCV(DecisionTreeClassifier(), {'max_depth': np.arange(10,50,5)})

# clf.fit(X_train, y_train)
# clf.best_estimator_
# clf.best_score_
# clf = GridSearchCV(DecisionTreeClassifier(), {'max_features': np.arange(5,35,5)})

# clf.fit(X_train, y_train)
# clf.best_params_
# clf.best_score_
test_submit_df.head()
X_test_submit = test_submit_df.values[:,1:]
X_test_submit = scaler.transform(X_test_submit)

X_test_submit_reduced = dimen_reduction.transform(X_test_submit)
submit_pred = lr.predict(X_test_submit_reduced)
submit_pred.shape[0]
pred_df = pd.DataFrame(list(zip(np.arange(submit_pred.shape[0]), submit_pred)), columns=['id','label'])
pred_df
pred_df.to_csv('submission.csv', index=False)