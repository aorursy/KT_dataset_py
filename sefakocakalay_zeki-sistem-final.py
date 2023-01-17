from sklearn.datasets import load_iris #data
#Gerekli Kütüphanelerin import edilmesi
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas as pd

df = load_iris()
X_train, X_test, y_train, y_test = train_test_split(df['data'], df['target'], test_size = 0.20, random_state=7)
#SVM with default(rbf) kernel
svm_model = SVC(random_state = 7).fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#SVM with linear kernel
svm_model = SVC(kernel ="linear",random_state = 7).fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#SVM with poly kernel
svm_model = SVC(kernel = "poly", random_state = 7).fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#Decision Tree Classifier with default parameters
cart = DecisionTreeClassifier(random_state = 7)
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#Decision Tree Classifier with different parameters
cart = DecisionTreeClassifier(max_depth =14, max_features='log2', random_state = 7)
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#yükseltemedim hocam bunları :D
#Decision Tree Classifier with different parameters-2
cart = DecisionTreeClassifier(max_depth = 3,max_features = 4, random_state = 7)
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#yükseltemedim hocam bunları :D
#K-nearest neighbors with default parameters
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#K-nearest neighbors with different parameters-1
knn = KNeighborsClassifier(n_neighbors=14, algorithm='ball_tree')
knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#K-nearest neighbors with different parameters-2
knn = KNeighborsClassifier(n_neighbors = 8, weights = 'distance',leaf_size = 39, p = 150)
knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#Random Forest Classifier with default parameters
rf_model = RandomForestClassifier(random_state = 7).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#Ada Boost Classifier with default parameters
adb_model = AdaBoostClassifier(random_state = 7).fit(X_train, y_train)
y_pred = adb_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#Ada Boost Classifier with svc model
svc = SVC(probability=True,kernel='linear')
adb_model = AdaBoostClassifier(base_estimator = svc, random_state = 7).fit(X_train, y_train)
y_pred = adb_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 


#Bagging Classifier with default parameters
bag_model = BaggingClassifier(random_state = 7).fit(X_train, y_train)
y_pred = bag_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
#Bagging Classifier with svc model
svc = SVC(probability=True, kernel='linear')
bag_model = BaggingClassifier(base_estimator = svc, random_state = 7).fit(X_train, y_train)
y_pred = bag_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 

#Voting Classifer with SVC,DTree and KNN
Knn_clf = KNeighborsClassifier()
DTree_clf = DecisionTreeClassifier()
SVC_clf = SVC()
voting_clf = VotingClassifier(estimators=[('SVC', SVC_clf), ('DTree', DTree_clf), ('Knn', Knn_clf)], voting='hard')
voting_clf.fit(X_train, y_train)
eclf_model = voting_clf.fit(X_train, y_train)
y_pred = eclf_model.predict(X_test)
print("**Confusion Matrix**\n",confusion_matrix(y_test, y_pred))
print("**Accuracy Score**\n:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
