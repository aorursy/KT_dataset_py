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
data = pd.read_csv("/kaggle/input/titanic/train.csv")

train = data[["Survived","Pclass", "Sex", "Age", "Fare"]]

train.head()
sex = {"female": 1, "male":2}

train["Sex"] = train["Sex"].map(sex)
train.head()
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
train = train.dropna()
x = train.drop("Survived", axis=1)

y = train["Survived"]
scaler = StandardScaler()

scaler.fit(x)

x_scaled = scaler.transform(x)
tree = DecisionTreeClassifier(random_state=42)

kf = KFold(random_state=42, shuffle=True, n_splits=5)

parameters = {

'max_depth': np.arange(3, 15),

'max_features': np.linspace(0.3, 1, 8),

'min_samples_leaf': np.arange(2, 50, 1)

}

cv = GridSearchCV(tree, param_grid=parameters, cv=kf, iid=True)
cv.fit(x_scaled, y)
print(cv.best_score_)

print(cv.best_params_)
score=[]

for i in range(1,50):

    knn = KNeighborsClassifier(n_neighbors=i)

    kf = KFold(random_state=42, shuffle=True, n_splits=5)

    cv = cross_val_score(knn, x_scaled, y, cv=kf)

    score.append(cv.mean())
print(max(score))

print(score.index(max(score))+1)
knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_scaled, y)
knn.predict(x_scaled[:10])

test = [[1, 1, 26, 20.000],[1, 2, 4, 40.000],[2, 1, 26, 10.000],[3, 1, 26, 1.000]]

knn.predict(scaler.transform(test))