from IPython.display import IFrame
IFrame('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', width=300, height=200)
#import load_iris function from datasets module
from sklearn.datasets import load_iris
# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)
#print the iris data
print(iris.data)
#print the names of the four features
print(iris.feature_names)
#print integers representing the species of each observation
print(iris.target)
## print the encoding scheme for speies 0 = setosa, 1 = versicolor, 2= virginica
print(iris.target_names)
# check the types of the feature and response
print(type(iris.data))
print(type(iris.target))
# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print(iris.data.shape)
# check the shape of the response (single dimension matching the number of observations)
print(iris.target.shape)
# store feature matrix in "X"
X = iris.data

#W store response vector in "y"
y = iris.target
# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target
# print the shapes of x and y
print(X.shape)
print(y.shape)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)
knn.fit(X,y)
knn.predict([[3, 5, 4, 2]])
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)
# instantiate the model
knn = KNeighborsClassifier(n_neighbors=5)

#fit the model with data
knn.fit(X, y)

# predict the response for new observations
knn.predict(X_new)
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the estimator ( using default parameters)
logreg = LogisticRegression()

# Fit the model with data
logreg.fit(X, y)

# predict the responde for new observations
logreg.predict(X_new)
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response values for the observation in X 
logreg.predict(X)
# store the predicted response values
y_pred = logreg.predict(X)

# check how many predictions were grenerated
len(y_pred)
# compute classification accuracy for the logistic regresaion model
from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))
# print the shapes of X and y
print(X.shape)
print(y.shape)
# STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
# print the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)
# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)
# STEP 2 :  train the model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# STEP 3 : make predicitons on the testing set
y_pred = logreg.predict(X_test)

# compare actual response values (y_test) with preedicted response values (y_predict)
print(metrics.accuracy_score(y_test, y_pred))
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
# trying K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
#import Matplotlin ( scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the note book
%matplotlib inline

#plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing accuracy')
# instatiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=11)

# train the model with X and y ( not X_train and y_train)
knn.fit(X, y)

# make a prediction for an out-of sample observation
knn.predict([[3, 5, 4, 2]])