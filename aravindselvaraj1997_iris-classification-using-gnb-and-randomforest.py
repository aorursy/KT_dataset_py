import pandas as pd
iris = pd.read_csv('../input/iris/Iris.csv')
iris.head()
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
X.head()
Y = iris['Species']
Y.head()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)
print(y_pred)
accuracy = metrics.accuracy_score(y_pred, y_test)
print(accuracy)
# For reference I tried with the data which is available from the input data
test = gnb.predict([[4.7, 3.2, 1.3, 0.2]])
print(test)
xrf_train, xrf_test, yrf_train, yrf_test = train_test_split(X, Y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

rf = RandomForestClassifier()
rf.fit(xrf_train, yrf_train)
y_pred = rf.predict(xrf_test)
accuracy = metrics.accuracy_score(y_pred, yrf_test)
print(accuracy)
# For reference I tried with the data which is available from the input data
test = rf.predict([[6.4, 3.1, 5.5, 1.8]])
print(test)
