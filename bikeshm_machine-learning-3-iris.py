from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X =iris.data
Y = iris.target
X
Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)

print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train,Y_train)
prediction = my_classifier.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train,Y_train)
prediction = my_classifier.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))


from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
prediction = gaussian.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
prediction = logreg.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, Y_train)
prediction = svc.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))
# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
prediction = linear_svc.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(X_train, Y_train)
prediction = randomforest.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))

