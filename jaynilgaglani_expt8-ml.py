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
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

train_df = pd.read_csv("../input/train-clean.csv")

train_df.drop('PassengerId',axis=1,inplace=True)

train_df.info()
from sklearn.preprocessing import MinMaxScaler

ss = MinMaxScaler()

col = train_df.columns

for i in col:

    train_df[i] = ss.fit_transform(train_df[[i]])

    

train_df.describe()
X = train_df.drop('Survived', axis=1)

Y = train_df['Survived']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)

lr
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

lr.score(x_test,y_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
from sklearn.model_selection import GridSearchCV
# Create regularization penalty space

penalty = ['l1', 'l2']



# Create regularization hyperparameter space

C = np.logspace(0, 4, 15)

solver = ['lbfgs','newton-cg']

multi_class = ['auto','ovr']

class_weight = ['balanced',None]



# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty,class_weight = class_weight,solver=solver,multi_class=multi_class)





clf = GridSearchCV(lr, hyperparameters, cv=6, verbose=0)

best_model = clf.fit(x_train,y_train)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])

print('Best C:', best_model.best_estimator_.get_params()['C'])

print('Best class_weight:', best_model.best_estimator_.get_params()['class_weight'])

print('Best solver:', best_model.best_estimator_.get_params()['solver'])

print('Best multi_class:', best_model.best_estimator_.get_params()['multi_class'])



y_pred = best_model.predict(x_test)

best_model.score(x_test,y_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
# log = LogisticRegression(C=1.0, class_weight='balanced', dual=False, fit_intercept=True,

#                    intercept_scaling=1, l1_ratio=None, max_iter=100,

#                    multi_class='auto', n_jobs=None, penalty='l2',

#                    random_state=42, solver='liblinear', tol=0.0001, verbose=0,

#                    warm_start=False)

# log