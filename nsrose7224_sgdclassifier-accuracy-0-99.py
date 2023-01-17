# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/creditcard.csv")

df.head(10)
values = df.values

X = values[:, :-1]

y = values[:, -1]

scaler = StandardScaler()

X = scaler.fit_transform(X)

X[:2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = SGDClassifier(shuffle=True)

model.fit(X_train, y_train)

model.score(X_test, y_test)