import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Extract data 

col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']

pima = pd.read_csv('../input/diabetes.csv')

pima.columns = col_names

X = pima[['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age']]

y = pima.label
def svm_pred(X,y):

    from sklearn.preprocessing import StandardScaler

    from sklearn.grid_search import RandomizedSearchCV



    from sklearn import svm

    clf = svm.SVC()

    

    #Standardise the features

    import numpy as np

    scalar = StandardScaler()

    X = scalar.fit_transform(X)



    #Split data for cross validation

    from sklearn.cross_validation import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 7)

    X = X_train

    y = y_train



    #Prepare parameter grid for model tuning

    C_range = np.logspace(-2,10,13)

    gamma_range = np.logspace(-9,3,13)

    param_grid = dict(gamma = gamma_range,C=C_range)

    grid = RandomizedSearchCV(clf,param_grid,cv=5,n_jobs = 2,n_iter=10,random_state=42) 

    grid.fit(X,y)



    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    print(grid.best_estimator_)



    #Train the model with best parameters

    model = svm.SVC(C=grid.best_params_['C'], cache_size=200, class_weight=None, coef0=0.0,\

      decision_function_shape=None, degree=3, gamma=grid.best_params_['gamma'], kernel='rbf',\

      max_iter=-1, probability=False, random_state=None, shrinking=True,\

      tol=0.001, verbose=False)

    

    #Check Accuracy on training and testing data set

    from sklearn.cross_validation import cross_val_score

    from sklearn.metrics import accuracy_score

    scores = cross_val_score(model,X,y,cv=10,scoring='accuracy')

    print("Accuracy on training set: %0.2f%% (+/- %0.2f%%)" % (100*scores.mean(), 100*scores.std() * 2))

    model.fit(X,y)

    y_pred = model.predict(X_test)

    print("Accuracy on testing set: %0.2f%%" % (100*accuracy_score(y_test,y_pred)))

    return;
#Training model on initial data

svm_pred(X,y)
#Adding new features based on standard ranges

pima['bp_bin'] = np.where(pima.bp >= 90,1,0)

pima['glucose_bin'] = np.where(pima.glucose > 170 ,1,0)

pima['insulin_bin'] = np.where(pima.insulin > 166,1,0)

pima['bmi_bin'] = np.where(pima.bmi >= 25,1,0)

X = pima[['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','bp_bin','glucose_bin','insulin_bin','bmi_bin']]

y = pima.label
#Training model with new features

svm_pred(X,y)