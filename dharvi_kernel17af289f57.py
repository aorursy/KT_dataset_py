# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams

# figure size in inches

rcParams['figure.figsize'] = 11.7,8.27





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv('../input/irisdataset/Iris.csv')

iris.head()
iris.info()
iris.describe()
#countplot for each of the labels

sns.countplot(x='label',data=iris, palette = 'Blues')

plt.show()
plt.subplot(2,2,1)

sns.boxplot(x = 'petal_length',data = iris, palette ='Blues')



plt.subplot(2,2,2)

sns.boxplot(x = 'petal_width',data = iris, palette ='Blues')



plt.show()
plt.subplot(2,2,1)

sns.boxplot(x = 'sepal_length',data = iris, palette ='Blues')



plt.subplot(2,2,2)

sns.boxplot(x = 'sepal_width',data = iris, palette ='Blues')



plt.show()
#PETALS



plt.subplot(2,2,1)

sns.boxplot(x='label', y='petal_length', data=iris, palette ='Blues')



plt.subplot(2,2,2)

sns.boxplot(x='label', y='petal_width', data=iris, palette ='Blues')

plt.show()
#SEPALS



plt.subplot(2,2,1)

sns.boxplot(x='label', y='sepal_length', data=iris , palette ='Blues')



plt.subplot(2,2,2)

sns.boxplot(x='label', y='sepal_width', data=iris , palette ='Blues')

plt.show()
colors = ["windows blue", "amber", "greyish","dusty purple"]

sns.pairplot(iris, hue='label',palette = 'Blues')

plt.show()
sns.heatmap(iris.corr(),annot = True)

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
Y = iris['label']

X = iris.drop(['label'],axis = 1)
X.head()
Y.head()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state = 100)
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
logr = LogisticRegression()



#fit the model on training data

logr.fit(X_train,Y_train)



#predict values for X_test

y_pred = logr.predict(X_test)
accuracy  = metrics.accuracy_score(y_pred,Y_test)

print("Accuracy of the model is : ",accuracy)
fig, ax = plt.subplots()

ax.scatter(Y_test, y_pred)

ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)

ax.set_xlabel('Actual')

ax.set_ylabel('Predicted')

plt.show()