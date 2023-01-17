#from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import math
data = pd.read_csv("../input/zoo.csv")
data.head(5)
from sklearn.model_selection import train_test_split
all_x = data.iloc[:, 1:17]
all_y = data.iloc[:, 17]
print(all_x)
print(all_y)
X_train,X_test,y_train,y_test = train_test_split(all_x,all_y)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape) 
#clf = svm.SVC()
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
clf.predict(X_test[10:15])
y_test[10:15] #model predict correctly with accuracy of 88% using SVM but using decison tree clasifier the accurary 
#export model
from sklearn.externals import joblib
joblib.dump(clf, 'zoomodel.joblib')