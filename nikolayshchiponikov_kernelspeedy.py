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
data.head()
data.describe()
data.loc[0:4]
data[["Name","Age"]].loc[:3]
data["Sex"].value_counts()
total = data.shape[0]
#n_surv = data["Survived"].value_counts()[0]

#surv = data["Survived"].value_counts()[1]

n_surv, surv = data["Survived"].value_counts()

print(np.round(surv*100/total,2))
f_class = data["Pclass"].value_counts()[1]

print(np.round(f_class*100/total,2))
data["Pclass"].hist();
print(np.round(data["Age"].mean(),2))

print(data["Age"].median())
data.loc[0:4]
data["SibSp"].corr(data["Parch"])
names = data[data["Sex"]=="female"]["Name"]
def filter_names(name):

    if "Miss." in name:

        lst = name.split(" ")

        idx = lst.index("Miss.")

        return lst[idx+1]

    if "(" in name:

        idx = name.find("(")

        return name[idx+1:-1].split()[0]
f_names= names.apply(filter_names)
f_names.value_counts()
data.head()
pdata = data[["Survived", "Pclass", "Sex", "Age"]]

pdata.head()

names = {"male":1, "female":2}
pdata["Sex"] = pdata["Sex"].map(names)
print(pdata.shape)

pdata.dropna()

print(pdata.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, KFold

from sklearn.preprocessing import StandardScaler
pdata = pdata.dropna()
x = pdata.drop("Survived",axis=1)

y = pdata["Survived"]

scaler = StandardScaler()

scaler.fit(x)

x_scaled = scaler.transform(x)

score = []

for i in range(1,50):

    knn = KNeighborsClassifier(n_neighbors=i)

    clf = cross_val_score(knn,x_scaled,y,cv=5)

    score.append(clf.mean())
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(score)

print(max(score))

print(score.index(max(score))+1)
x_scaled[:,:10]
tree = DecisionTreeClassifier(random_state=42, max_depth=20, max_features=3)

clf = cross_val_score(tree,x_scaled,y,cv=5)

print(clf.mean())
# knn = KNeighborsClassifier(n_neighbors=6)

# knn.fit(x_scaled,y);

# knn.predict(scaler.transform([[1,2,25]]))