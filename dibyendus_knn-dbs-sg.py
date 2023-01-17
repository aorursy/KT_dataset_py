from sklearn import datasets

myiris = datasets.load_iris()

x = myiris.data

y = myiris.target

type(x)

x.shape

type(y)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

x_scaled = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)
#Accuracy

print('Accuracy = ', knn.score(x_test, y_test))



#Confusion Matrix

from sklearn.metrics import confusion_matrix

print('\nConfusion matrix')

print(confusion_matrix(y_test, y_pred))



#Classification Report

from sklearn.metrics import classification_report

print('\nClassification Report')

print(classification_report(y_test, y_pred))  