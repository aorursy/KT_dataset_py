#importing the modules we need

import sklearn

from sklearn import datasets

from sklearn import svm

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
#loading dataset from sklearns very own datasets module

cancer = datasets.load_breast_cancer()
#train data

x = cancer.data



#test data

y = cancer.target
cancer.feature_names
#splitng data

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)



classes = ['malignant', 'benign']



#model

clf = svm.SVC(kernel='linear',C=1)

clf.fit(x_train,y_train)



y_pred = clf.predict(x_test)



acc = metrics.accuracy_score(y_test, y_pred)

#accuracy of the prediction

acc