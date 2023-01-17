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
import matplotlib.pyplot as plt

plt.style.use("ggplot")
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head()
X = df.drop("target", axis=1)

y = df['target']
y.hist(bins=2)
X.head()
X.describe()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape)

print(X_test.shape)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train, y_train)
preds = rfc.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy_score(y_test, preds)
print(classification_report(y_test, preds))
confusion_matrix(y_test, preds)