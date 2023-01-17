import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import random 
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
#from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE
#import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
alldata=pd.read_csv("../input/creditcard.csv")
#alldata.head()

#label i/p o/p class
X = alldata.drop('Class',axis=1)
y = alldata.Class

#show the no of records and positive examples  
all_records= len(alldata)
number_records_fraud = len(alldata[alldata.Class == 1])
print(all_records,number_records_fraud)

#SMOTE sampling

X_resample, y_resample = SMOTE().fit_sample(X, y)
print ('The number of transactions after resampling : ' + str(len(X_resample)))
print ('If the number of frauds is equal to the number of normal tansactions? '+ str(sum(y_resample == 0) == sum(y_resample == 1)))

X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size=0.3, random_state=0)


# Logistic Regression\n",

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_test,y_test) * 100, 2)
print(acc_log)

# svc = SVC()
# svc.fit(X_train,y_train)
# y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_test,y_test) * 100, 2)
# print(acc_svc)

# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(X_train,y_train)
# y_pred = knn.predict(X_test)
# acc_knn = round(knn.score(X_test,y_test) * 100, 2)
# print(acc_knn)

# gaussian = GaussianNB()
# gaussian.fit(X_train,y_train)
# y_pred = gaussian.predict(X_test)
# acc_gaussian = round(gaussian.score(X_test,y_test) * 100, 2)
# acc_gaussian

# perceptron = Perceptron()
# perceptron.fit(X_train,y_train)
# y_pred = perceptron.predict(X_test)
# acc_perceptron = round(perceptron.score(X_test,y_test) * 100, 2)
# acc_perceptron

# linear_svc = LinearSVC()
# linear_svc.fit(X_train,y_train)
# y_pred = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_test,y_test) * 100, 2)
# acc_linear_svc

# sgd = SGDClassifier()
# sgd.fit(X_train,y_train)
# y_pred = sgd.predict(X_test)
# acc_sgd = round(sgd.score(X_test,y_test) * 100, 2)
# acc_sgd

# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train,y_train)
# y_pred = decision_tree.predict(X_test)
# acc_decision_tree = round(decision_tree.score(X_test,y_test) * 100, 2)
# print(acc_decision_tree)

# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train,y_train)
# y_pred = random_forest.predict(X_test)
# random_forest.score(X_test,y_test)
# acc_random_forest = round(random_forest.score(X_test,y_test) * 100, 2)
# print(acc_random_forest)


# models = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron',
#        'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree'],'Score': [acc_svc, acc_knn, acc_log,acc_random_forest, acc_gaussian, acc_perceptron,
#        acc_sgd, acc_linear_svc, acc_decision_tree]})
# models.sort_values(by='Score', ascending=False)
