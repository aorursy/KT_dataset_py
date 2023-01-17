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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv("../input/advertising.csv")
data.head()
data.info()
data.describe()
data.columns
data.head()
data.info()
data.describe()
sns.distplot(data["Daily Time Spent on Site"])
sns.distplot(data["Age"])
sns.distplot(data["Area Income"])
sns.distplot(data["Daily Internet Usage"])
data.corr()
sns.heatmap(data.corr(), annot = True, cmap = "Greens")
sns.boxplot(data["Male"],data[ "Daily Time Spent on Site"], palette = "BuPu")
sns.boxplot(data["Clicked on Ad"], data ["Age"], palette= "Greens")
sns.scatterplot(data["Daily Time Spent on Site"], data["Daily Internet Usage"])
sns.scatterplot(data["Daily Time Spent on Site"], data["Area Income"])
sns.boxplot(data["Clicked on Ad"], data["Daily Time Spent on Site"], palette= "BuGn_r")
sns.boxplot(data["Clicked on Ad"], data["Age"], palette= "BuGn_r")
sns.boxplot(data["Clicked on Ad"], data["Area Income"], palette= "BuGn_r")
sns.boxplot(data["Clicked on Ad"], data["Male"], palette= "BuGn_r")
sns.jointplot(x = "Age", y = "Area Income",  data = data)
sns.jointplot(x = "Age", y = "Daily Time Spent on Site",  data = data,color = "green" , kind = "kde")
sns.pairplot(data)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
X1 = data[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage","Male" ]]
y1 = data ["Clicked on Ad"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.4, random_state = 123) 
log_model = LogisticRegression()

log_model.fit(X1_train, y1_train)
prediction1 = log_model.predict(X1_test)
print(classification_report(y1_test, prediction1))
from sklearn import metrics
metrics.accuracy_score(y1_test, prediction1)
X2 = data[["Daily Time Spent on Site", "Age", "Area Income"]]
y2 = data["Clicked on Ad"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.4, random_state = 123)
log_model2 = LogisticRegression()
log_model2.fit(X2_train, y2_train)
prediction2 = log_model2.predict(X2_test)
metrics.accuracy_score(y2_test, prediction2)