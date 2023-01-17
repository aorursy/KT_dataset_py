# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/" + 'train.csv')

train.head()
train.info()

train["Age"].mode()
plt.hist(train.Age, train.Survived)
import matplotlib.pyplot as plt

plt.bar(train.Embarked, train.Survived)



from sklearn.tree import DecisionTreeClassifier

test = pd.read_csv("../input/" + 'test.csv')

clf = DecisionTreeClassifier()

y = clf.fit(train)

print(y.predict(test))

