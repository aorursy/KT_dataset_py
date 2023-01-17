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
dataset=pd.read_csv("../input/iris-dataset/iris.csv")
dataset.head()
#cheking for the null values and replacing them with the na values
dataset.isnull().sum()
#luckly we do not have any null values
#now we can describe the data
dataset.describe()
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
#making scatter plot by class
classes=dataset['species'].unique()
print("Different calsses---names")
print(classes)

sns.pairplot(dataset,hue='species')
plt.figure(figsize=(10, 10))

for column_index, column in enumerate(dataset.columns):
    if column == 'species':
        continue
    plt.subplot(2, 2, column_index + 1)
    sns.violinplot(x='species', y=column, data=dataset)
from sklearn.model_selection import train_test_split
labels = dataset["species"].values
new = dataset.drop(columns=["species"])
X_train,X_test,Y_train,Y_test = train_test_split(new,labels,test_size=0.25,random_state=42)
X_train.head()
Y_train[:5]
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,Y_train)
lr.predict(X_test)
lr.score(X_test,Y_test)
#definately it is overfitting we can visualize the data to see the overfitting
from sklearn import svm
from sklearn.metrics import accuracy_score
svc = svm.SVC()
svc.fit(X_train,Y_train)
y_pred=svc.predict(X_test)
accuracy_score(Y_test,y_pred)

