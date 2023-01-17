# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/TitanicDataset/titanic_data.csv")

data = data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])

data['Sex'] = data['Sex'].factorize()[0]

#data['Cabin'] = data['Cabin'].factorize()[0]

data['Embarked'] = data['Embarked'].factorize()[0]

data = data.dropna()

survivors = data.loc[data['Survived']==1]

died = data.loc[data['Survived']==0]

X = data.copy().drop(columns=['Survived'])

Y = data['Survived']

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, Y)
data
plt.scatter(survivors['Age'], survivors['Pclass'], c="green")

plt.scatter(died['Age'], died['Pclass'], c="red")

plt.show()
data
clf = MultinomialNB()

clf.fit(train_x, train_y)
predictions = clf.predict(test_x)
from sklearn.metrics import accuracy_score

accuracy_score(test_y, predictions)