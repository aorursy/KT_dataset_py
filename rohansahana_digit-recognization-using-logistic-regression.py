import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        
from sklearn.preprocessing import StandardScaler           
from sklearn.linear_model import LogisticRegression       

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/digit-recognizer/train.csv')
data
# sns.heatmap(data.isnull(), cbar = False, cmap= 'gist_rainbow_r')
X = data.iloc[:,1:].values        
Y = data.iloc[:,0].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)
X_train = X
Y_train = Y
# FS = StandardScaler()                   #Feature Scaling and is applied to Independent variables or features of data
# X_train = FS.fit_transform(X_train)     
# X_test = FS.fit_transform(X_test)       

# #Created a function with many Machine Learning Models
# def models(X_train,Y_train):

#   #Using Logistic Regression Algorithm to the Training Set
#   from sklearn.linear_model import LogisticRegression
#   log = LogisticRegression(random_state = 0)
#   log.fit(X_train, Y_train)
 
#   #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
#   from sklearn.neighbors import KNeighborsClassifier
#   knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#   knn.fit(X_train, Y_train)

#   #Using SVC method of svm class to use Support Vector Machine Algorithm
#   from sklearn.svm import SVC
#   svc_lin = SVC(kernel = 'linear', random_state = 0)
#   svc_lin.fit(X_train, Y_train)

#   #Using SVC method of svm class to use Kernel SVM Algorithm
#   from sklearn.svm import SVC
#   svc_rbf = SVC(kernel = 'rbf', random_state = 0)
#   svc_rbf.fit(X_train, Y_train)

#   #Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
#   from sklearn.naive_bayes import GaussianNB
#   gauss = GaussianNB()
#   gauss.fit(X_train, Y_train)

#   #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
#   from sklearn.tree import DecisionTreeClassifier
#   tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#   tree.fit(X_train, Y_train)

#   #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
#   from sklearn.ensemble import RandomForestClassifier
#   forest = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
#   forest.fit(X_train, Y_train)
 
#   #printing model accuracy on the training data.
#   print('Logistic Regression Training Accuracy: ', log.score(X_train, Y_train))
#   print('K Nearest Neighbor Training Accuracy: ', knn.score(X_train, Y_train))
#   print('Support Vector Machine (Linear Classifier) Training Accuracy: ', svc_lin.score(X_train, Y_train))
#   print('Support Vector Machine (RBF Classifier) Training Accuracy: ', svc_rbf.score(X_train, Y_train))
#   print('Gaussian Naive Bayes Training Accuracy: ', gauss.score(X_train, Y_train))
#   print('Decision Tree Classifier Training Accuracy: ', tree.score(X_train, Y_train))
#   print('Random Forest Classifier Training Accuracy: ', forest.score(X_train, Y_train))
#   return log, knn, svc_lin, svc_rbf, gauss, tree, forest

# # Executing the above function
# model = models(X_train,Y_train)

# from sklearn.metrics import confusion_matrix 
# for i in range(len(model)):
#   cm = confusion_matrix(Y_test, model[i].predict(X_test))   #extracting TN, FP, FN, TP
#   TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()
#   print(cm)
#   print('Model[{}] Testing Accuracy = "{} !"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))  # Calculating testing accuracy
#   print()  # Print a new line since its a by default feature of print function
classifierLR = LogisticRegression(random_state= 0, max_iter=200000)    #Defining Model
classifierLR.fit(X_train, Y_train)
Y_pred = classifierLR.predict(X_test)
score = classifierLR.score(X_train, Y_train)*100
print("Score = {}".format(score))
dte = pd.read_csv("../input/digit-recognizer/test.csv")
# dte.dropna(inplace = True)

X_TEST = dte.values

# FST = StandardScaler()
# X_TEST = FST.fit_transform(X_TEST)
# X_TEST

Y_PRED = classifierLR.predict(X_TEST)
Predicted = pd.DataFrame(Y_PRED)
ss = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
dataset = pd.concat([ss['ImageId'], Predicted], axis=1)
dataset.columns = ['ImageId', 'Label']
dataset.to_csv('sample_submission-LR.csv', index = False)