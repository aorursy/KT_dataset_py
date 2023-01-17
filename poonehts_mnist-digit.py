import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')

train.head()
X_test=test/255.0

X_test.head()
# load the MNIST dataset

y = train.label

X = train.drop(['label'],axis=1)/255.0

X.head()

y.head()
from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier as dtc

from sklearn.neighbors import KNeighborsClassifier as knnc

from sklearn.naive_bayes import GaussianNB as gnb



from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
#model = rfc()

#cv_scores=cross_val_score(model, X, y, cv=5)

#print('mean acuuracy:',cv_scores.mean())



classifiers = [rfc(),dtc(),SVC(),SVC(kernel='linear'),gnb(),knnc()]



classifier_names = ['Random Forest',

                    'Decision Tree Classifier',

                    'SVM classifier with RBF kernel',

                    'SVM classifier with linear kernel',

                    'Gaussian Naive Bayes',

                   'K nearest neighbors']



for clf, clf_name in zip(classifiers, classifier_names):

    cv_scores = cross_val_score(clf, X, y, cv=5)

    

    print(clf_name, ' mean accuracy: ', round(cv_scores.mean()*100, 3), '% std: ', round(cv_scores.var()*100, 3),'%')
import time



time_start = time.time()



model = knnc(n_neighbors=5)



cv_scores=cross_val_score(model, X, y, cv=5)



print('mean acuuracy:',cv_scores.mean())



print('K nearest neighbors is done! Time elapsed: {} seconds'.format(time.time()-time_start))
from sklearn.model_selection import RandomizedSearchCV

import time



param_dist = {'n_neighbors':[5,10,15],

              'leaf_size':[5,10,15]}



time_start = time.time()



random_search = RandomizedSearchCV(estimator=knnc(), 

                           param_distributions=param_dist,

                           n_iter=30,                     

                           scoring='accuracy',

                           cv=3)



random_search.fit(X, y)



print('Time elapsed: {} seconds'.format(time.time()-time_start))
print(random_search.best_params_)

print("\n",random_search.best_estimator_)
from sklearn.metrics import accuracy_score



knn_clf=knnc(n_neighbors= random_search.best_params_['n_neighbors'],

                                leaf_size= random_search.best_params_['leaf_size'])





cv_scores=cross_val_score(knn_clf, X, y, cv=5)

print('mean acuuracy:',cv_scores.mean())



knn_clf.fit(X,y)
# Evaluate the model on the test set

y_hat_knn=knn_clf.predict(X_test)
output = pd.DataFrame({'ImageId': X_test.index+1, 'Label': y_hat_knn})
output.head()
output.to_csv('my_submission2.csv', index=False)

print("Your submission was successfully saved!")