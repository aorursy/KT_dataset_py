import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

data=pd.read_csv('../input/forestfires.csv')

data.head()
y=data.temp

x=data.drop('temp',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

x_train.head()
x_train.shape
x_test.head()
x_test.shape
iris=load_iris()

x,y=iris.data, iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.5, test_size=0.5, random_state=123)

y_test
y_train

from sklearn.linear_model import LinearRegression as lm

model=lm().fit(x_train,y_train)

predictions=model.predict(x_test)

import matplotlib.pyplot as plt

plt.scatter(y_test,predictions)

plt.xlabel('True values')

plt.ylabel('Predictions')

plt.show()