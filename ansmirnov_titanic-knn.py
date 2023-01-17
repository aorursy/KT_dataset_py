import numpy as np 

import pandas as pd

from sklearn import neighbors
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
def sex_to_int(x):

    if x == 'male':

        return 1

    else:

        return 0



def to_int(x):

    try:

        return int(x)

    except:

        return 0

    

X = pd.concat([train.Pclass.map(to_int), train.Age.map(to_int), train.Sex.map(sex_to_int)], axis=1)

Y = train.Survived

X.head()
clf = neighbors.KNeighborsClassifier()

clf.fit(X, Y)
X = pd.concat([test.Pclass.map(to_int), test.Age.map(to_int), test.Sex.map(sex_to_int)], axis=1)

res = pd.DataFrame(test.PassengerId)

res.insert(1, 'Survived', clf.predict(X))

res.to_csv('titanic.csv',index=False)