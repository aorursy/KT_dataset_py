import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')

data.shape
data.columns
data.isnull().sum()
## Viz



plt.figure(figsize=[8,6])

sns.heatmap(data.corr(), annot=True)
sns.pairplot(data, hue='target_class')
## Scaling



X = data.drop(['target_class'], axis=1)

y = data['target_class']



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



X_scaled = MinMaxScaler(feature_range=(0,1))

X_train_scaled = X_scaled.fit_transform(X)



X_train_scaled



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
## Models



from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import mean_squared_error



rfr = RandomForestClassifier()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier()

nb = GaussianNB()

lr = LogisticRegression()
#RFR 

rfr.fit(X_train, y_train)

prediction = rfr.predict(X_test)

score_train = round(rfr.score(X_train, y_train)*100)

score_test = round(rfr.score(X_test, y_test)*100)

report = classification_report(y_test, prediction)



print('Train Accuracy: %s' %score_train), '%'

print('Test Accuracy: %s' %score_test), '%', '\n'

print('Classification report: '), '\n'

print(report)



# metrics



mse = round(mean_squared_error(y_test, prediction), 4)

rmse = round(np.sqrt(mse), 4)

print ('Root Mean Squared Error: %s' %rmse)
# KNN



knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

score_train = round(knn.score(X_train, y_train)*100)

score_test = round(knn.score(X_test, y_test)*100)

report = classification_report(y_test, prediction)



print('Train Accuracy: %s' %score_train), '%'

print('Test Accuracy: %s' %score_test), '%', '\n'

print('Classification report: '), '\n'

print(report)



# metrics



mse = round(mean_squared_error(y_test, prediction), 4)

rmse = round(np.sqrt(mse), 4)

print ('Root Mean Squared Error: %s' %rmse)
## DTC



dtc.fit(X_train, y_train)

prediction = dtc.predict(X_test)

score_train = round(dtc.score(X_train, y_train)*100)

score_test = round(dtc.score(X_test, y_test)*100)

report = classification_report(y_test, prediction)



print('Train Accuracy: %s' %score_train), '%'

print('Test Accuracy: %s' %score_test), '%', '\n'

print('Classification report: '), '\n'

print(report)



# metrics



mse = round(mean_squared_error(y_test, prediction), 4)

rmse = round(np.sqrt(mse), 4)

print ('Root Mean Squared Error: %s' %rmse)
## GNB



nb.fit(X_train, y_train)

prediction = nb.predict(X_test)

score_train = round(nb.score(X_train, y_train)*100)

score_test = round(nb.score(X_test, y_test)*100)

report = classification_report(y_test, prediction)



print('Train Accuracy: %s' %score_train), '%'

print('Test Accuracy: %s' %score_test), '%', '\n'

print('Classification report: '), '\n'

print(report)



# metrics



mse = round(mean_squared_error(y_test, prediction), 4)

rmse = round(np.sqrt(mse), 4)

print ('Root Mean Squared Error: %s' %rmse)
## Logistic Regression



lr.fit(X_train, y_train)

prediction = lr.predict(X_test)

score_train = round(lr.score(X_train, y_train)*100)

score_test = round(lr.score(X_test, y_test)*100)

report = classification_report(y_test, prediction)



print('Train Accuracy: %s' %score_train), '%'

print('Test Accuracy: %s' %score_test), '%', '\n'

print('Classification report: '), '\n'

print(report)



# metrics



mse = round(mean_squared_error(y_test, prediction), 4)

rmse = round(np.sqrt(mse), 4)

print ('Root Mean Squared Error: %s' %rmse)