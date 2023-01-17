# Importing packages



import numpy as np 

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt 
# Getting and analysing the data



data = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')



data.isnull().sum()
data.head()
data.describe()
# Plotting the data



plt.subplot(2, 1, 1)

sns.scatterplot(x = 'sepal_length', y = 'sepal_width', data = data, hue = 'species')



plt.subplot(2, 1, 2)

sns.scatterplot(x = 'petal_length', y = 'petal_width', data = data, hue = 'species')



plt.show()
# Time to use some machine learning algortihms

# Lets split the species from the whole dataset and try predicting using the basic algorithms



from sklearn.model_selection import train_test_split

from sklearn import preprocessing 

  

# Since models require numbers rather than strings to fit a model we need to convert the species column

encoder = preprocessing.LabelEncoder() 

  

data['species']= encoder.fit_transform(data['species']) 



features = data.drop(['species'], axis = 1)

actual = data['species']



# Time to split data using sklearn



X_train, X_test, Y_train, Y_test = train_test_split(features, actual, test_size = 0.3, random_state = 5)



X_train.head()

Y_train.head()
# Lets try a simple Logistic regression



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression(solver = 'lbfgs', multi_class='auto')

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

print(accuracy_score(Y_test, Y_pred))
# Lets Implement a simple k nearest neighbors also known as knn model



from sklearn.neighbors import KNeighborsClassifier



k_range = list(range(1,10))

scores = {}

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)

    scores.update({str(k) : accuracy_score(Y_test, Y_pred)})

   

print(scores)
# Lets Implement a simple k nearest neighbors also known as knn model



from sklearn.tree import DecisionTreeClassifier



DecisionTreeClas = DecisionTreeClassifier()

DecisionTreeClas.fit(X_train, Y_train)

Y_pred = DecisionTreeClas.predict(X_test)

print(accuracy_score(Y_pred, Y_test))