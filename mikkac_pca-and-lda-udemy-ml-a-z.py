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
df = pd.read_csv('/kaggle/input/wine-customer-segmentation/Wine.csv')

df.head()
X = df.iloc[:, :-1]

X.head()
y = df.iloc[:, -1:]

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)

# X_train = pca.fit_transform(X_train)

# X_test = pca.transform(X_test)

# explained_variance = pca.explained_variance_ratio_

# X_train[:, :5]
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)

X_train = lda.fit_transform(X_train, y_train)

X_test = lda.transform(X_test)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42).fit(X_train, y_train)



y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score



ac = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print(ac)

print(cm)