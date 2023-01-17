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
data["Sex"].value_counts()
total = data.shape[0]

print(total)
n_alive, alive = data["Survived"].value_counts()

p_alive = alive*100/total

print(n_alive, alive)

print(np.round(p_alive,2))
fc_pass = data["Pclass"].value_counts()[1]

p_fclass = fc_pass*100/total

print(np.round(p_fclass,2))
data["Pclass"].hist();
print(data["Age"].mean())

print(data["Age"].median())

data["SibSp"].corr(data["Parch"])
numeric = data[["PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare"]].corr()
import seaborn as sns

%matplotlib inline
sns.heatmap(numeric, annot=True);
names = data[data["Sex"]=="female"]["Name"]
def filter_names(name):

    if "Miss." in name:

        lst = name.split(" ")

        idx = lst.index("Miss.")

        return(lst[idx+1])

    if "(" in name:

        idx = name.find("(")

        return(name[idx+1:-1].split(" ")[0])

#    print(name)
names_chart = names.apply(filter_names).value_counts()
names_chart
data.head()
pdata = data[["Survived","Pclass","Sex","Age"]]

pdata.head()

names = {"male":1, "famale":2}
pdata["Sex"] = pdata["Sex"].map(names)


print(pdata.shape)

pdata.dropna()

print(pdata.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, KFold
pdata = pdata.dropna()
x = pdata.drop("Survived",axis=1)

y = pdata["Survived"]

score = []

for i in range(1,50):

    knn = KNeighborsClassifier(n_neighbors=i)

    clf = cross_val_score(knn,x,y,cv=5)

    score.append(clf.mean())
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(score)

print(max(score))

print(score.index(max(score))+1)

knn = KNeighborsClassifier(n_neighbors=13)

knn.fit(x,y);

knn.predict(x.loc[:10])