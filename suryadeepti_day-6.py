from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv("https://raw.githubusercontent.com/suryadeepti/Naive_bayes_data/master/Naive-Bayes-Classification-Data.csv")

data.head()
x=data.drop('diabetes', axis=1)

y=data['diabetes']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25, random_state=40)
from sklearn.naive_bayes import GaussianNB

model=GaussianNB()

model.fit(x_train,y_train)
plt.plot(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
plt.plot(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)*100

accuracy