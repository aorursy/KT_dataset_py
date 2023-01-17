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
dataset = pd.read_csv(os.path.join(dirname,filenames[1]))
dataset.head()
print(len(dataset),' ',dataset.shape)
dataset.info()
dataset.count()
dataset.head(2)

iris=dataset

import matplotlib.pyplot as plt
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()

dataset.drop('Id',axis=1,inplace=True)
from sklearn.model_selection import train_test_split 
train , test = train_test_split(dataset,test_size=0.3)
print(train.shape,' ', test.shape)
train_x = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_y = train.Species
test_x = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

test_y = test.Species
from sklearn import svm
model = svm.SVC()

model.fit(train_x,train_y)
y_pred=model.predict(test_x)
from sklearn import metrics

metrics.accuracy_score(y_pred,test_y)