import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
X = train.drop(['Outcome'], axis = 1)
y = train.Outcome
# parameters = {'criterion': ('gini', 'entropy'), 'n_estimators': [10, 50, 100, 105, 150]}
# gb = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf = GridSearchCV(gb, parameters)
clf.fit(X,y)
predicted = clf.predict(test)
print(predicted)
output = pd.DataFrame(predicted,columns = ['Outcome'])
test = pd.read_csv('../input/test.csv')
output['Id'] = test['Id']
output[['Id','Outcome']].to_csv('submission.csv', index = False)
output.head()