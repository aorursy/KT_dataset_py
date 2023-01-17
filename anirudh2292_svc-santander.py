# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import numpy as np # linear algebra

import pandas as pd



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.metrics import precision_recall_fscore_support as score



from sklearn.pipeline import Pipeline



from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score



from time import time



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report





import matplotlib.pyplot as plt

from pylab import rcParams

import seaborn as sb





print(os.listdir("../input"))
san_train = pd.read_csv('../input/train.csv')

print(san_train)
X = san_train.drop(['target','ID_code'], axis=1)



y = san_train['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=17)
print(sorted(sklearn.metrics.SCORERS.keys()))
pipeline = Pipeline([

('clf', SVC(kernel='rbf', gamma = 0.01, C=100))

])



parameters = {

'clf__gamma': (0.01,0.03,0.1,0.3,1),

'clf__C': (0.1,0.3,1,3,10,30),

'clf__kernel':('linear','poly')

}



grid_search = GridSearchCV(pipeline,parameters,n_jobs=2,verbose=1,scoring='accuracy')

grid_search.fit(X_train[:1000],y_train[:1000])

print('Best score: %0.3f' % grid_search.best_score_)

print('Best parameters set:')

best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(best_parameters.keys()):

    print('\t%s: %r' % (param_name,best_parameters[param_name]))

predictions = grid_search.predict(X_test)

print(classification_report(y_test,predictions))
san_test = pd.read_csv('../input/test.csv')

print(san_test)
y_pred_orig = grid_search.predict(san_test[X_train.columns])

y_pred_final =pd.DataFrame(y_pred_orig,columns=["target"])

df_out = pd.concat([san_test.ID_code,y_pred_final], axis=1 )

df_out.to_csv('SAN_SVC.csv', index=False)