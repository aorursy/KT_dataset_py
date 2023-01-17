# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df.head(5)
sample = sample.reshape(28,28)
plt.matshow(sample, cmap = 'gray')
X = df.drop("label" , axis = 1 )

y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
print(f1_score(y_test, predictions, average="weighted"))

print("Accuracy:", accuracy_score(y_test, predictions))
rf_model.predict(X_test.iloc[15].values.reshape(1, -1))
y_test[15]
plt.matshow(X_test.iloc[15].values.reshape(28, 28), cmap="gray")