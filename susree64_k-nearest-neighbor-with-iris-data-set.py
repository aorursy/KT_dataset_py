## import standard libraries for our work
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import sklearn
from sklearn.datasets import load_iris
iris = load_iris()
type(iris)
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
data1.shape  #There are 5 features ( columns ) and 150 rows, observations
data1.head()
data1['target'].value_counts()
### There are basically 3 types of categories 0 means setosa, 1 means versicolor, and 2 means virginica
def categorize(a):
    if a == 0.0:
        return('setosa')
    if a == 1.0:
        return('versicolor')
    return('virginica')
data1['Species'] = data1['target'].apply(categorize)
    
data1.head()
plt.figure(figsize=[18,8])
plt.scatter(data1['Species'], data1['sepal length (cm)'],  marker= 'o')
plt.scatter(data1['Species'], data1['sepal width (cm)'], marker= 'x')
plt.scatter(data1['Species'], data1['petal width (cm)'], marker= '*')
plt.scatter(data1['Species'], data1['petal length (cm)'], marker= ',')
plt.ylabel('Length in cm')
plt.legend()
plt.xlabel('Species Name')
plt.show()
plt.figure(figsize=[18,8])
plt.plot(data1['sepal length (cm)'], marker= 'o')
plt.plot(data1['sepal width (cm)'], marker= 'x')
plt.plot(data1['petal length (cm)'], marker= '*')
plt.plot(data1['petal width (cm)'], marker= ',')
plt.ylabel('Length in cm')
plt.legend()
plt.show()
sns.jointplot(data1['sepal length (cm)'], data1['sepal width (cm)'], size= 13, kind = 'kde')

sns.jointplot(data1['petal length (cm)'], data1['petal width (cm)'], size= 13, kind = 'kde')
### It is a standard convention to name X_train in capital X and y_train in small letters. 
###  All the measurements (features) are considered as X and the Species is considered as y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data1[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']], data1['Species'], random_state=0 )

X_train.head()
y_train.head()
X_test.head()
y_test.head()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train) # This is fitting the model  with the training data. 
prediction = knn.predict(X_test) # By supplying the test data now predicting the  Y (Species values)
prediction
y_test + "  " +  prediction #
#Comparision of the predicted data from the Test sent and the y_test data
# Predicted data and the y_test data are same. This gives the highest confidence level on the model built
### Now we can test the model using any data and it would be accurate 

X_new = np.array([[5, 2.9, 1, 0.2]])
predection1 = knn.predict(X_new)
predection1

