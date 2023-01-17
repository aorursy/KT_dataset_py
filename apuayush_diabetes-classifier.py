import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

%matplotlib inline 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import mpld3 as mpl


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


import xgboost as xgb
data = pd.read_csv('../input/diabetes.csv')
data.head()
print(len(data))
data.hist(figsize=(10,8))
import seaborn as sns

ax = sns.countplot(data.iloc[:,-1],label="Count") 
corr = data[data.columns].corr()
sns.heatmap(corr, annot = True)
train, test = train_test_split(data, test_size = 0.25, random_state = 2)
X_train, Y_train = train.iloc[:,:-1].values, train.iloc[:,-1]
X_test, Y_test = test.iloc[:,:-1].values, test.iloc[:,-1]
def classification_model(model, data, predictors, outcome):
  model.fit(data[predictors], data[outcome])
  predictions = model.predict(data[predictors])
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 
indp_var=list(data.columns[0:-1])
outcome_var = 'Outcome'
model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=6)
classification_model(model,train,indp_var,outcome_var)
classification_model(model, test, indp_var, outcome_var)
import _pickle as cPickel
with open('diabetes.pickle', 'wb') as f:
    cPickel.dump(model, f)
dtrain = xgb.DMatrix(X_train, Y_train)
dtest = xgb.DMatrix(X_test, Y_test)
param = {'max_depth' : 6, 'eta' : 0.1, 'objective' : 'binary:logistic', 'seed' : 42}
num_round = 50
bst = xgb.train(param, dtrain, num_round, [(dtest, 'test'), (dtrain, 'train')])
preds = bst.predict(dtest)
preds[preds > 0.5] = 1
preds[preds <= 0.5] = 0
print(accuracy_score(preds, Y_test), 1 - accuracy_score(preds, Y_test))
clf = SVC(kernel="linear", C=0.1)
clf.fit(X_train, Y_train)
y_eval = clf.predict(X_test)
print(accuracy_score(y_eval, Y_test))
