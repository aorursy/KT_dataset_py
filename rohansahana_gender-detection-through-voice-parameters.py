import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dtr = pd.read_csv('../input/gender-voice-detection/train.csv')

# dtr.drop(labels = ['mindom', 'median'], axis = 1, inplace= True)

dtr.info()
X = dtr.iloc[:,1:21].values

Y = dtr.iloc[:,21].values
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.95, random_state = 0)



X_train = X

Y_train = Y
from sklearn.preprocessing import StandardScaler

FS = StandardScaler()

X_train = FS.fit_transform(X_train)

X_test = FS.fit_transform(X_test)
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

#   forest = RandomForestClassifier(n_estimators=100,random_state = 0)

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
#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100,random_state = 0)

classifier.fit(X_train, Y_train)
#Get the importance of the features

forest = classifier[6]

importances = pd.DataFrame({'feature':dtr.iloc[:, 1:21].columns,'importance':np.round(forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances
Y_pred = classifier.predict(X_test)   # Prediction was made from test data which we got due to splitting
dte = pd.read_csv("../input/gender-voice-detection/test.csv")   # Read the test dataset provided to us

# dte.drop(labels = ['mindom', 'median'], axis = 1, inplace= True)

dte
X_TEST = dte.iloc[:,1:].values           # Same procedure is to be followed as it was followed in case of training dataset

FST = StandardScaler()

X_TEST = FST.fit_transform(X_TEST)

Y_PRED = classifier.predict(X_TEST)   # Final Data was predicted
Predicted = pd.DataFrame(Y_PRED)   # New Dataframe was created

ss = pd.read_csv("../input/gender-voice-detection/sample-submission.csv")

dataset = pd.concat([ss['Id'], Predicted], axis=1)     # Concatenation (Merging) with sample-submission dataset and creating a new dataset

dataset.columns = ['Id', 'label']      # Column names were assigned for new dataset formed

dataset.to_csv('sample_submission-rf.csv', index = False)   # Exported the new dataset with name