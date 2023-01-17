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
cover = pd.read_csv("../input/train.csv")

cover.info()



cover['Age'].fillna(20,inplace=True)

cover.info()

print(cover["Cabin"].mode())
cover["Cabin"].fillna("B96 B98", inplace=True)

cover.info()
print(cover["Embarked"].mode())
cover["Embarked"].fillna("S", inplace=True)

cover.info()
mapping = {0:"mati", 1:"hidup"}

cover["Class"]= cover["Survived"].map(mapping)
cover.info()
del cover["Survived"]
cover.info()
cover
del cover["PassengerId"]

del cover["Name"]
cover
cover.info()
X = cover[cover.columns[0:9]]

Y = cover["Class"]

olist = list(X.select_dtypes(["object"]))



X
def normalize(lst):

    s = sum(lst)

    return map(lambda x: float(x)/s, lst)
cover
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors=5, weights='uniform')

model.fit(X,Y)

model.score(X,Y)


