import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

iris =pd.read_csv("../input/iris/Iris.csv") 

iris.head()
iris.drop(["Id"],axis=1,inplace=True)
iris.head()
sns.heatmap(data=iris.corr(),annot=True)

plt.show()
iris.hist(figsize=(35,20),edgecolor="black")
# Importing all the Machine Learning Model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn import svm
train, test = train_test_split(iris,test_size=0.3)

print("The shape of test: {}". format(test.shape))

print("The shape of train: {}". format(train.shape))
test_X = test[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

train_X = train[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

test_y = test.Species

train_y = train.Species

le = LabelEncoder()

train_y = le.fit_transform(train_y)

test_y = le.fit_transform(test_y)



#Logictic Regression

lr = LogisticRegression()

lr.fit(train_X,train_y)

pred_lr = lr.predict(test_X)

print("The accuracy of the model:{}".format(metrics.accuracy_score(pred_lr,test_y)*100))

# Random Forest Regression

rfr = RandomForestRegressor()

rfr = rfr.fit(train_X,train_y)

pred_rfr = rfr.predict(test_X)

print("The accuracy of the model:{}".format(metrics.accuracy_score(pred_rfr.round(),test_y)*100))



# support vector machine

sv_m = SVR()

sv_m =sv_m.fit(train_X,train_y)

pred_svm = sv_m.predict(test_X)

print("The accuracy of the model:{}".format(metrics.accuracy_score(pred_svm.round(),test_y)*100))





dtc = DecisionTreeClassifier()

dtc.fit(train_X,train_y)

pred_dtc = dtc.predict(test_X)

print("The accuracy of the model:{}".format(metrics.accuracy_score(pred_dtc,test_y)*100))


