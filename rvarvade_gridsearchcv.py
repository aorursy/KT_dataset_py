from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits



digits = load_digits()

dir(digits)
from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np



X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2)

len(X_train)
len(X_test)
model = svm.SVC(kernel = 'rbf', C = 30 , gamma = 'auto')

model.fit(X_train,y_train)

model.score(X_test,y_test)
from sklearn.model_selection import cross_val_score



cross_val_score(svm.SVC(kernel = 'rbf', C = 30 , gamma = 'auto'),digits.data,digits.target,cv=5)
cross_val_score(svm.SVC(kernel = 'linear', C = 10 , gamma = 'auto'),digits.data,digits.target,cv=5)
cross_val_score(svm.SVC(kernel = 'linear', C = 20 , gamma = 'auto'),digits.data,digits.target,cv=5)
kernels = ['rbf','linear']

C = [1,10,20]



avg_scores = {}



for kval in kernels:

    for cval in C:

        cv_scores = cross_val_score(svm.SVC(kernel = kval, C = cval , gamma = 'auto'),digits.data,digits.target,cv=5)

        avg_scores[kval+'_'+str(cval)] = np.average(cv_scores)

        

avg_scores
from sklearn.model_selection import GridSearchCV



clf = GridSearchCV(svm.SVC(gamma = 'auto'),{

    'C':[1,10,20],

    'kernel': ['rbf','linear']

}, cv = 5, return_train_score=False)



clf.fit(digits.data,digits.target)

clf.cv_results_
df = pd.DataFrame(clf.cv_results_)

df
df[['param_C','param_kernel','mean_test_score']]
dir(clf)
clf.best_params_
clf.best_score_
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd



rs = RandomizedSearchCV(svm.SVC(gamma = 'auto'),{

    'kernel':['rbf','linear'],

    'C':[1,10,20]

}, cv=5, return_train_score=False,

                       n_iter=4)



rs.fit(digits.data,digits.target)

pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier



model_params = {

    

    'svm': {

        'model': svm.SVC(gamma='auto'),

        'params': {

            'C':[1,10,20],

            'kernel':['rbf','linear']

        }

    },

    'random_forest':{

        'model': RandomForestClassifier(),

        'params':{

            'n_estimators':[1,5,10,20]

        }

    },

    'logistic_regrssion':{

        'model': LogisticRegression(solver='liblinear',multi_class='auto'),

        'params':{

            'C':[1,5,10]

        }

    },

    'naive_bayes':{

        'model': GaussianNB(),

        'params':{

            

        }

    },

    'naive_bayes':{

        'model': MultinomialNB(),

        'params':{

            

        }

    },

    'decision_tree':{

        'model': DecisionTreeClassifier(),

        'params':{

            'criterion':['gini','entropy'],

            'splitter':['best','random']

            

        }

    }

}
scores = []



for model_name, mp in model_params.items():

    clf = GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)

    clf.fit(digits.data,digits.target)

    scores.append({

        'model':model_name,

        'best_score':clf.best_score_,

        'best_params':clf.best_params_

    })
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

df