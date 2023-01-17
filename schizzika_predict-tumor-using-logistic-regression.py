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
import pandas as pd
df = pd.read_csv("/kaggle/input/breast_cancer.csv")
df.head()
X = df.iloc[:, 1:-1].values

y = df["Class"].values
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state = 0)

model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
model.score(X_test, y_predicted)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)

print(cm)
(84+47)/(84+47+3+3)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)

print("Accuracy {:.2f}%".format(accuracies.mean() * 100))

print("Standard Deviation {:.2f}%".format(accuracies.std() * 100))


