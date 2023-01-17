#Import lib 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn import datasets 
from sklearn import svm

#Loading Dataset 
iris = datasets.load_iris() 
print(iris.data.shape,iris.target.shape) 
print ("Iris data set Description : ", iris['DESCR']) 
#Create training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0) 
print (X_train.shape, y_train.shape) 
print (X_test.shape, y_test.shape) 
#Using SVM classifier 
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train) 
print(clf.score(X_test, y_test))
#Cross fit metrics 
scores = cross_val_score(clf, iris.data, iris.target, cv=5) 
print(scores) 
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
