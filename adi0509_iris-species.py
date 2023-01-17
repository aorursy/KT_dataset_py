import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt



#from sklearn.datasets import load_iris

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
iris = pd.read_csv("../input/iris/Iris.csv")

iris.head()
iris.info()
iris.describe()
#snsdata = iris.drop(['Id'], axis=1)

g = sns.pairplot(iris, hue='Species', markers='x')

g = g.map_upper(plt.scatter)

g = g.map_lower(sns.kdeplot)
sns.set_style("whitegrid");

sns.pairplot(iris, hue="Species", height=3);
encode = LabelEncoder()

iris.Species = encode.fit_transform(iris.Species)

iris.head()
iris.head(-5)
# train-test-split   

train , test = train_test_split(iris,test_size=0.2,random_state=0)



print('shape of training data : ',train.shape)

print('shape of testing data',test.shape)
train.head()
test.head()
# seperate the target and independent variable

train_x = train.drop(columns=['Species'],axis=1)

train_y = train['Species']



test_x = test.drop(columns=['Species'],axis=1)

test_y = test['Species']
model = LogisticRegression()



model.fit(train_x,train_y)



predict = model.predict(test_x)



print('Predicted Values on Test Data',encode.inverse_transform(predict))



print('\n\nAccuracy Score on test data : \n\n')

print(accuracy_score(test_y,predict))