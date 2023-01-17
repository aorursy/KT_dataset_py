# required libraries



import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

# read the dataset

iris_data = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

iris_data.head()
iris_data.tail()
iris_data.columns
iris_data.columns = iris_data.columns.str.title()
iris_data.columns
encode = LabelEncoder()

iris_data.Species = encode.fit_transform(iris_data.Species)

iris_data.head()
iris_data.describe()
iris_data.info()
sns.set_style("whitegrid")

sns.pairplot(iris_data, hue="Species")
sns.heatmap(iris_data.corr(), cmap="magma",annot=True)

from sklearn.model_selection import train_test_split
X = iris_data.drop('Species', axis=1)

y = iris_data['Species']
# train-test-split   



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

print('shape of training data : ',X_train.shape)

print('shape of testing data',X_test.shape)





# create the object of the model

model = LogisticRegression(solver='newton-cg', multi_class='auto')



model.fit(X_train,y_train)



predict = model.predict(X_test)

predict
from sklearn.metrics import classification_report,confusion_matrix
# Summary of the predictions made by the classifier

print(classification_report(y_test, predict))

print(confusion_matrix(y_test, predict))



# Accuracy score

print('\n\nAccuracy Score on test data : \n\n')

print(accuracy_score(y_test,predict))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
# NOW WITH K=23

knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=23')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))