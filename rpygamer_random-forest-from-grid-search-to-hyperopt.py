# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#=====================================================================================================================

import pandas as pd                                       # data processing and CSV file I/O library

import numpy as np                                        # Algebra library

#import pandas_profiling as pdp                            # explore data 

#======================================================================================================================

from sklearn.ensemble.forest import RandomForestClassifier# to import the random forest Model 

from sklearn.metrics import roc_curve, auc                # to import roc curve abd auc metrics for evaluation 

from sklearn.grid_search import GridSearchCV              # grid search is used for hyperparameters-optimization

from sklearn.model_selection import KFold                # cross validation using the kfold algorithm

#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials     # library for hyperparameters-optimization

#=======================================================================================================================

#  Plotting imports

import seaborn as sns                                     # Python graphing library

import matplotlib.pyplot as plt

%matplotlib inline
def Performance(Model,Y,X):

    # Perforamnce of the model

    fpr, tpr, _ = roc_curve(Y, Model.predict_proba(X)[:,1])

    AUC  = auc(fpr, tpr)

    print ('the AUC is : %0.4f' %  AUC)

    plt.figure()

    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % AUC)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.legend(loc="lower right")

    plt.show()
frame= pd.read_csv("../input/train.csv")

frame.head()
data=frame.drop(['Name','Ticket','PassengerId'],axis=1)

data.head()
data=data.fillna(-9999,inplace=False)
BM=pd.get_dummies(data)
BM.head()
# split the farme data to train and validation datasets 70/30.

from sklearn.model_selection import train_test_split

train, valid = train_test_split(BM, test_size = 0.3,random_state=1991)
train.shape
valid.shape
train=pd.DataFrame(train,columns=BM.columns)

valid=pd.DataFrame(valid,columns=BM.columns)

X_train=train.drop(['Survived'],axis=1)

Y_train=train['Survived']

X_valid=valid.drop(['Survived'],axis=1)

Y_valid=valid['Survived']
X_train.head()
X_valid.head()
RF0=RandomForestClassifier()
RF0.fit(X=X_train,y=Y_train)
Performance(Model=RF0,Y=Y_valid,X=X_valid)
grid_1 = { "n_estimators"      : [100,200,500],

               "criterion"         : ["gini", "entropy"],

               "max_features"      : ['sqrt','log2',0.2,0.5,0.8],

               "max_depth"         : [3,4,6,10],

               "min_samples_split" : [2, 5, 20,50] }
RF=RandomForestClassifier()

grid_search = GridSearchCV(RF, grid_1, n_jobs=-1, cv=5)

grid_search.fit(X_train, Y_train)
grid_search.grid_scores_
grid_search.best_score_
grid_search.best_params_
Performance(Model=grid_search,Y=Y_valid,X=X_valid)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score

def acc_model(params):

    clf = RandomForestClassifier(**params)

    return cross_val_score(clf, X_train, Y_train).mean()



param_space = {

    'max_depth': hp.choice('max_depth', range(1,20)),

    'max_features': hp.choice('max_features', range(1,150)),

    'n_estimators': hp.choice('n_estimators', range(100,500)),

    'criterion': hp.choice('criterion', ["gini", "entropy"])}



best = 0

def f(params):

    global best

    acc = acc_model(params)

    if acc > best:

        best = acc

    print ('new best:', best, params)

    return {'loss': -acc, 'status': STATUS_OK}



trials = Trials()

best = fmin(f, param_space, algo=tpe.suggest, max_evals=100, trials=trials)

print ('best:')

print (best)
RF2=RandomForestClassifier(max_features=113, n_estimators=498, criterion= 'entropy', max_depth=2,random_state=1)
RF2.fit(X=X_train,y=Y_train)
Performance(Model=RF2,Y=Y_valid,X=X_valid)