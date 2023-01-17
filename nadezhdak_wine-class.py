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
from sklearn.datasets import load_wine
wine_data = load_wine()
data = pd.DataFrame(wine_data["data"], columns=wine_data["feature_names"])

data["Target"] = wine_data["target"]
data.head()
from sklearn.tree import DecisionTreeClassifier
x = data.drop("Target", axis=1)

y = data["Target"]
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x, y)
print(clf.predict([x.loc[90]]))

print(data.loc[90])
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(random_state=42, max_depth=3, max_features=3)

cross_val = cross_val_score(clf, x, y, cv=5)

print(cross_val.mean())
print(cross_val)
from sklearn.preprocessing import StandardScaler
skl = StandardScaler()

skl.fit (x)

x_scaled = skl.transform(x)
x_scaled.shape
clf = DecisionTreeClassifier(random_state=42)

model = cross_val_score(clf, x, y, cv=5)

print(model.mean())

model = cross_val_score(clf, x_scaled, y, cv=5)

print(model.mean())
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
score = []

for i in range(1, 80):

    knn = KNeighborsClassifier(n_neighbors=i)

    cv = KFold (random_state=42, n_splits=5, shuffle=True)

    res = cross_val_score(knn, x_scaled, y, cv=cv)

    score.append(res.mean())
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(score);

plt.grid()

print(max(score))

print(score.index(max(score))+1)
knn = KNeighborsClassifier(n_neighbors=29)

res = cross_val_score(knn, x_scaled, y, cv=5)

score.append(res.mean())

print(res.mean())
knn.fit(x_scaled, y);

knn.predict([x_scaled[98]])