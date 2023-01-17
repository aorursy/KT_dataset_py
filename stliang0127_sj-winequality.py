# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib 

import matplotlib.pyplot as plt
dt_1 = pd.read_csv('/kaggle/input/wine-quality-dataset/train.csv')

print("Print first 5 subjects")

print(dt_1.head())

print("")

print("Basic descriptive statistics for all features")

print(dt_1.describe())

print("")

print("Feaure attributes")

print(dt_1.info())
dt_white = dt_1[dt_1['kind']=='white']

dt_red = dt_1[dt_1['kind']=='red']
def Hist_p(dataset):

    dataset.iloc[:, 1:].hist(bins=20, figsize=(20, 10))

    plt.show()

    

print('White wine histogram')    

Hist_p(dt_white)

print(' ')    

print('Red wine histogram')    

Hist_p(dt_red)
print('White wine correlation map') 

corr_white = dt_white.corr()

corr_white.style.background_gradient(cmap='coolwarm')

print('Red wine correlation map') 

corr_red = dt_red.corr()

corr_red.style.background_gradient(cmap='coolwarm')
White_X = dt_white.iloc[:, 2:13]

White_y = dt_white.iloc[:, 13]



Red_X = dt_red.iloc[:, 2:13]

Red_y = dt_red.iloc[:, 13]



#print(White_X.head())

#print(White_y.head())

#print(Red_X.head())

#print(Red_y.head())
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



def Dtree(X, y, label):

    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)

    scores = cross_val_score(clf, X, y, cv=5)

    print(label)

    print("Decision Tree Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #print(scores.mean())

    #Default score for Decision Tree: Mean accuracy of self.predict(X) wrt. y. 

    #accuracy = # of correct / # of prediction



def Rfrst(X, y, label):    

    clrf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

    scores = cross_val_score(clrf, X, y, cv=5)

    print(label)

    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #Default score for Random Forest: Mean accuracy of self.predict(X) wrt. y. 

Dtree(White_X, White_y, "White wine")

Rfrst(White_X, White_y, "White wine")
Dtree(Red_X, Red_y, "Red wine")

Rfrst(Red_X, Red_y, "Red wine")
from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 250, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model



def RANDOM_T(X, y):

    rf_random.fit(X, y)

    
RANDOM_T(White_X, White_y)

rf_random.best_params_
clrf_grid = RandomForestClassifier(n_estimators=63, min_samples_split=5, min_samples_leaf= 2, max_features= 'auto', max_depth=70, random_state=0, bootstrap= False)

scores = cross_val_score(clrf_grid, White_X, White_y, cv=5)

print("White wine Grid Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Default score for Random Forest: Mean accuracy of self.predict(X) wrt. y. 
RANDOM_T(Red_X, Red_y)
rf_random.best_params_
clrf_grid = RandomForestClassifier(n_estimators=250, min_samples_split=5, min_samples_leaf= 1, max_features= 'sqrt', max_depth=100, random_state=0, bootstrap= True)

scores = cross_val_score(clrf_grid, Red_X, Red_y, cv=5)

print("Red wine Grid Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Default score for Random Forest: Mean accuracy of self.predict(X) wrt. y. 
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
xgb = XGBClassifier()

#kfold = KFold(n_splits=5, random_state=7)

def XGBs(X, y, label):

    results = cross_val_score(xgb, X, y, cv=5)

    print(label)

    print("XGBoost Accuracy: %.2f (+/- %.2f)" % (results.mean(), results.std()* 2))
XGBs(White_X, White_y, "White wine")

XGBs(Red_X, Red_y, "Red wine")
dt_test = pd.read_csv('/kaggle/input/wine-quality-dataset/test.csv')

print("Print first 5 subjects")

print(dt_test.head())
dtt_white = dt_test[dt_test['kind']=='white']

dtt_red = dt_test[dt_test['kind']=='red']

dtt_white_X=dtt_white.iloc[:, 2:13]

#dtt_white_X.head()

dtt_red_X=dtt_red.iloc[:, 2:13]
#for white

clrf_gridw = RandomForestClassifier(n_estimators=63, min_samples_split=5, min_samples_leaf= 2, max_features= 'auto', max_depth=70, random_state=0, bootstrap= False)

clrf_gridw.fit(White_X, White_y)

result_white = clrf_gridw.predict(dtt_white_X)



#for red

clrf_gridr = RandomForestClassifier(n_estimators=250, min_samples_split=5, min_samples_leaf= 1, max_features= 'sqrt', max_depth=100, random_state=0, bootstrap= True)

clrf_gridr.fit(Red_X, Red_y)

result_red = clrf_gridw.predict(dtt_red_X)
White_submitt=pd.DataFrame({"Id": dtt_white['Id'], "quality": result_white})

Red_submitt=pd.DataFrame({"Id": dtt_red['Id'], "quality": result_red})
SJ_submit = pd.concat([White_submitt, Red_submitt]).sort_index()

SJ_submit

pd.DataFrame(SJ_submit).to_csv("submit_SJ.csv", index=False)