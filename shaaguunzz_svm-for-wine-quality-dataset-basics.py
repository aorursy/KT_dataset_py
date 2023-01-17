
#importing all the modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#importing the data
data = pd.read_csv("../input/wine-quality/winequalityN.csv")
data.head()
data.info()
data.describe()
#to check if there is null values
data.isnull().sum()
#filling null value using mean
mean = data['fixed acidity'].mean()
data['fixed acidity'].fillna(mean,inplace=True)

mean = data['volatile acidity'].mean()
data['volatile acidity'].fillna(mean,inplace=True)


mean = data['citric acid'].mean()
data['citric acid'].fillna(mean,inplace=True)
mean = data['residual sugar'].mean()
data['residual sugar'].fillna(mean, inplace=True)
mean = data['chlorides'].mean()
data['chlorides'].fillna(mean,inplace=True)
mean = data['pH'].mean()
data['pH'].fillna(mean,inplace=True)
mean = data['sulphates'].mean()
data['sulphates'].fillna(mean,inplace=True)
data.isnull().sum()

#let's start visualizing
plt.figure(figsize=(20,10))
sns.boxplot(data=data, palette="Set3")
plt.show()

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = data)

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = data)
sns.pairplot(data, hue=  'type')
#since some variables needs to be encoded
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
data1= data.drop('quality', axis =1).copy()
data1= data1.apply(LabelEncoder().fit_transform)
sclr = StandardScaler().fit(data1)
X=sclr.transform(data1)
y= data['quality']
#splitting the data set
X_train,X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=1)
svc = SVC().fit(X,y)
print("Accuracy on training set: {:.4f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.4f}".format(svc.score(X_test, y_test)))
y_pred =svc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred ))
#now finding best parameter using Grid Search CV
from sklearn.model_selection import GridSearchCV
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
#now let's try with best params
svc = SVC(C=1.3, gamma= 1.1, kernel ='rbf')
svc.fit(X, y)
print("Accuracy on training set: {:.4f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.4f}".format(svc.score(X_test, y_test)))
y_pred =svc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred ))
#however i found that 
svc = SVC(kernel='rbf',C=10,gamma=1)
svc.fit(X, y)
print("Accuracy on training set: {:.4f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.4f}".format(svc.score(X_test, y_test)))
