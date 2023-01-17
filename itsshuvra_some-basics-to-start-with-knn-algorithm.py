import numpy as np

a = np.array([[1,2,3],[4,5,6]])

print("a:\n{}".format(a))
from scipy import sparse

#eye function makes diagonal value 1 and others 0

dia1 = np.eye(4)

print("Numpy Array:\n",dia1)

#find the index of non zero values by using csr

spaMat = sparse.csr_matrix(dia1)

print("Scipy sparse CSR matrix:\n",spaMat)
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100) #-10 to 10 and 100 = total points in between

y = np.sin(x)

plt.plot(x,y,marker='+')
import pandas as pd

#dataset

ds = {

    'Name': ['Rodrygo', 'Kroos', 'Valverde', 'Isco'],

    'Location': ['Brazil', 'Germany', 'Uruguay', 'Spain'],

    'Age': ['18', '30', '21','28']

}

df = pd.DataFrame(ds)

display(df) #display all the data

display(df[df.Name =='Isco']) #select the specific one
from sklearn.datasets import load_iris

iris_dataset = load_iris()

print("Keys:\n",iris_dataset.keys())

print("\nFeature Names:\n",iris_dataset['feature_names'][:193])

print("\nTarget Names:\n",iris_dataset['target_names'])

print("\nData Shape:\n",iris_dataset['data'].shape)

print("\nData(first five cloumns):\n",iris_dataset['data'][:5])

print("\nTarget:\n",iris_dataset['target'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X train shape:\n", X_train.shape)

print("y train:\n", y_train[:5])
!pip install mglearn
from pandas.plotting import scatter_matrix

import mglearn 

irisdf = pd.DataFrame(X_train, columns= iris_dataset.feature_names)

#display(X_train[:5])

#display(irisdf)

# create a scatter matrix

gr = scatter_matrix(irisdf, c=y_train, marker='o',figsize=(15,15),hist_kwds={'bins':20},s=60, alpha=.8, cmap=mglearn.cm3)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)
#create a new data with np array

X_new = np.array([[8,4.5, 4, 4.8]]) #scikit always expect 2d array

print("Shape of new data: ",X_new.shape)

#make a prediction for this new data

prediction = knn.predict(X_new)

print("This is the data of-",iris_dataset['target_names'][prediction])
y_pred = knn.predict(X_test)

print(iris_dataset['target_names'][y_pred])

print("test data accuracy: ",np.mean(y_pred==y_test))