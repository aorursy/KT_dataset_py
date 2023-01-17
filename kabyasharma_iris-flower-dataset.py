# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris =pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
iris
iris['species'].value_counts()
iris.info()
iris.head()
iris.groupby('species').mean()
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Bar chart to show how sepal length differs with different species")

# Bar chart showing relation between sepal length and species
sns.barplot(x=iris['species'], y=iris['sepal_length'])

# Add label for vertical axis
plt.ylabel("sepal lenghth")
# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Bar chart to show how petal length differs with different species")

# Bar chart showing relation between sepal length and species
sns.barplot(x=iris['species'], y=iris['petal_length'])

# Add label for vertical axis
plt.ylabel("petal lenghth")
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=iris)
# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=iris.corr(), annot=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
encode = LabelEncoder()
iris.species = encode.fit_transform(iris.species)

print(iris)
# train-test-split   
train , test = train_test_split(iris,test_size=0.16,random_state=0)

print('shape of training data : ',train.shape)
print('shape of testing data',test.shape)
# seperate the target and independent variable
train_i = train.drop(columns=['species'],axis=1)
train_o = train['species']

test_i= test.drop(columns=['species'],axis=1)
test_o = test['species']
train_i.head()


train_o.head()
# LogisticRegression
classifier = LogisticRegression()
classifier.fit(train_i,train_o)

y_pred = classifier.predict(test_i)
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,test_o))
# Support Vector Machine's 
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(train_i, train_o)

y_pred = classifier.predict(test_i)
print(classification_report(test_o, y_pred))
print('accuracy is',accuracy_score(y_pred,test_o))
