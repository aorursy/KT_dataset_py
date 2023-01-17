# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")



data.drop(["Unnamed: 32","id"], axis=1,inplace=True)

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

print(data.info())

y= data.diagnosis.values

x_data = data.drop(["diagnosis"], axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15, random_state=42)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("score:",dt.score(x_test, y_test))



y_pred= dt.predict(x_test)

y_true= y_test
from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_true, y_pred)
import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True, linewidths=0.5, linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_head")

plt.ylabel("y_true")

plt.show()