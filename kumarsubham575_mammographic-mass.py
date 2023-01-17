# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.


from matplotlib import pyplot as plt

import seaborn as sns

# Load the dataset

data = pd.read_csv("../input/Cleaned_data.csv")



# Print the first few entries of the data

data.head()
data.describe()
plt.plot(data['BI-RADS'])
data.info()
data.isnull().sum()
data.dtypes
sns.barplot(data['BI-RADS'], data['Age'])


plt.rcParams['figure.figsize'] = (15, 5)

sns.countplot(data['Age'])
sns.jointplot(data['Severity'], data['Shape'])
sns.violinplot(data['Severity'], data['Density'])
sns.boxplot(data['Severity'], data['Margin'])
# splitting into x and y



x = data.iloc[:, :-1]

y = data.iloc[:, -1]
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.decomposition import PCA



pca = PCA(n_components = None)



x_train = pca.fit_transform(x_train)

x_test = pca.transform(x_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model = LogisticRegression()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



print("Testing Accuracy :", model.score(x_test, y_test))

print("Training Accuracy :", model.score(x_train, y_train))



# confusion matrix

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot = True)



# classification report

cr = classification_report(y_pred, y_test)

print(cr)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model = LogisticRegression()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



print("Testing Accuracy :", model.score(x_test, y_test))

print("Training Accuracy :", model.score(x_train, y_train))



# confusion matrix

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot = True)



# classification report

cr = classification_report(y_pred, y_test)

print(cr)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model = LogisticRegression()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



print("Testing Accuracy :", model.score(x_test, y_test))

print("Training Accuracy :", model.score(x_train, y_train))



# confusion matrix

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot = True)



# classification report

cr = classification_report(y_pred, y_test)

print(cr)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model = RandomForestClassifier()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



print("Testing Accuracy :", model.score(x_test, y_test))

print("Training Accuracy :", model.score(x_train, y_train))



# confusion matrix

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot = True)



# classification report

cr = classification_report(y_pred, y_test)

print(cr)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model = AdaBoostClassifier()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



print("Testing Accuracy :", model.score(x_test, y_test))

print("Training Accuracy :", model.score(x_train, y_train))



# confusion matrix

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot = True)



# classification report

cr = classification_report(y_pred, y_test)

print(cr)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model = DecisionTreeClassifier()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



print("Testing Accuracy :", model.score(x_test, y_test))

print("Training Accuracy :", model.score(x_train, y_train))



# confusion matrix

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot = True)



# classification report

cr = classification_report(y_pred, y_test)

print(cr)