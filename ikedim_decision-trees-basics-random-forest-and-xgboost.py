# load libraries:

import numpy as np

import pandas as pd

from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score, KFold, train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

import xgboost as xgb

# load the sklearn diabetes data set:

diabetes = load_diabetes()

X,y = diabetes.data,diabetes.target

rs = 32 # Random state initializer for repeatable results

X, test_X, y, test_y = train_test_split(X,y,test_size=0.2,random_state=rs)

diabetes_featureNames = 'age sex bmi bp tc ldl hdl tch ltg glu'.split()

np.set_printoptions(suppress=True,precision=5)

print('Diabetes training set:',X.shape[0],'examples,',X.shape[1],'features')

print('First example:')

for fName,f in zip(diabetes_featureNames,X[0]) :

    print('{}: {:.4f}'.format(fName,f))

print('y:',y[0])

def printRFScores(r,X,y,msg=None) :

    "Fits a random forest model, and prints training and out-of-bag score."

    r.fit(X,y)

    if msg is not None :

        print(msg,end=' ')

    print('Training score: {:.4f}  OOB score: {:.4f}'.format(r.score(X,y),r.oob_score_))



# rescale a couple of the input features:

print('Checking sensitivity to scaling:')

XRescaled = X.copy()

XRescaled[:,2] *= 100.0  # multiply by 100

XRescaled[:,4] = ((XRescaled[:,4]+100.0)**2)  # add 100 and square it

print('          Original   Rescaled')

for fName,oFeat,mFeat in zip(diabetes_featureNames,X[0],XRescaled[0]) :

    print('{:>4} {:>12.4f} {:>10.4f}'.format(fName,oFeat,mFeat))



# compare scores with original and rescaled features:

r = RandomForestRegressor(n_estimators=20,oob_score=True,random_state=rs)

printRFScores(r,X,y,'Original features:')

r = RandomForestRegressor(n_estimators=20,oob_score=True,random_state=rs)

printRFScores(r,XRescaled,y,'Rescaled features:')

def print_feature_importances(r,featureNames) :

    for featureName,importance in zip(featureNames,r.feature_importances_) :

        print('{:>3}: {:.4f}'.format(featureName,10.0*importance))

print_feature_importances(r,diabetes_featureNames)
def test_RFRs(X,y,kwList) :

    for kws in kwList :

        r = RandomForestRegressor(**dict(kws,oob_score=True,random_state=rs))

        printRFScores(r,X,y,str(kws)+':')

test_RFRs(X,y,[dict(n_estimators=n) for n in [20,40,60,100,200,300,400,500]])
test_RFRs(X,y,[dict(n_estimators=300,max_features=n) for n in range(1,11)])
def printMAE(r,X,y) :

    y_pred = r.predict(X)

    print('{:.3f}'.format(mean_absolute_error(y,y_pred)))

r = RandomForestRegressor(n_estimators=300,max_features=3,random_state=rs)

r.fit(X,y)

printMAE(r,X,y)

printMAE(r,test_X,test_y)

# split off a separate validation set:

train_X, val_X, train_y, val_y = train_test_split(X,y,test_size=0.2,random_state=rs)



def tryXGB(learning_rate, n_estimators=1000, early_stopping_rounds=5, verbose=False) :

    r = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

    r.fit(train_X, train_y, early_stopping_rounds=early_stopping_rounds,

          eval_set=[(val_X, val_y)], verbose=verbose)

    printMAE(r,X,y)



for i in range(20) :

    learning_rate = (i+1)*0.01

    print('{:.2f}'.format(learning_rate),end=' ')

    tryXGB(learning_rate)
tryXGB(0.13,verbose=True)
r = xgb.XGBRegressor(n_estimators=38, learning_rate=0.13)

r.fit(X, y)

printMAE(r,X,y)

printMAE(r,test_X,test_y)
