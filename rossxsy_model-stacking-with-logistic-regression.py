import numpy as np

import scipy as sp

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn



%matplotlib inline



# from sklearn.preprocessing import LabelEncoder

# from sklearn.cross_validation import StratifiedShuffleSplit



train = pd.read_csv('../input/train.csv').drop('id',axis=1)

test = pd.read_csv('../input/test.csv')

test_ids = test['id']

test.drop('id',axis=1,inplace=True)
print(train.isnull().any().any())

print(test.isnull().any().any())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
species = train['species']

train.drop('species',axis=1,inplace=True)

y_train = le.fit_transform(species)
from sklearn.preprocessing import MaxAbsScaler
x_data = np.vstack([train,test])

mas = MaxAbsScaler()

n_x_data = mas.fit_transform(x_data)

print(n_x_data.shape)

n_x_data
x_test = n_x_data[len(species):,:]

x_train = n_x_data[0:len(species),:]
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import accuracy_score, log_loss
seed=1

models = [

            'ADB',

            'GBC',

            'RFC',

            'KNC',

            'SVC'

         ]

clfs = [

        AdaBoostClassifier(random_state=seed,n_estimators = 150, learning_rate = 0.01), # best score -2.27

        GradientBoostingClassifier(random_state=seed,min_samples_split=2, n_estimators=100, learning_rate=0.01, 

                                   max_depth=3, min_samples_leaf=4), # best score -2.13

        RandomForestClassifier(random_state=seed,n_jobs=-1,min_samples_split=2,n_estimators=100,

                               criterion='gini',min_samples_leaf=1),

        KNeighborsClassifier(n_jobs=-1,n_neighbors=5, weights='distance', leaf_size=15),

        SVC(random_state=seed,probability=True,kernel='sigmoid',C=100, tol=0.005)

        ]
pred_train_models = []

pred_test_models = []
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import StratifiedKFold



kfold = 3 # use a bigger number



sss = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)

cvfolds = list(sss.split(x_train,y_train))



for j,clf in enumerate(clfs):

    print(j,clf)

    dataset_test_j = 0 

    dataset_train_j = np.zeros((x_train.shape[0],len(np.unique(y_train))))

    for i,(train_index, test_index) in enumerate(cvfolds):

        n_x_train, n_x_val = x_train[train_index], x_train[test_index]

        n_y_train, n_y_val = y_train[train_index], y_train[test_index]

        print('fold ' + str(i))        

        clf.fit(n_x_train,n_y_train)

        dataset_train_j[test_index,:] = clf.predict_proba(n_x_val)

        dataset_test_j += clf.predict_proba(x_test)

    pred_train_models.append(dataset_train_j)

    pred_test_models.append(dataset_test_j/float(kfold))

    

pred_blend_train = np.hstack(pred_train_models)

pred_blend_test = np.hstack(pred_test_models)
print('\Blending results with a Logistic Regression ... ')



blendParams = {'C':[1000],'tol':[0.01]} # test more values in your local machine

clf = GridSearchCV(LogisticRegression(solver='newton-cg', multi_class='multinomial'), blendParams, scoring='log_loss',

                   refit='True', n_jobs=-1, cv=5)

clf.fit(pred_blend_train,y_train)

print('The Best parameters of the blending model\n{}'.format(clf.best_params_))

print('The best score:{}'.format(clf.best_score_))



estimates = clf.predict_proba(pred_blend_test)

submission = pd.DataFrame(estimates, index=test_ids, columns=le.classes_)

submission.to_csv('./blendedEnsembles.csv')