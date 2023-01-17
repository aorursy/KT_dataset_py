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
human_train=pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/train.csv')
human_test=pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/test.csv')
x_test=human_test.drop(['subject','Activity'],axis=1)

y_test=human_test.Activity
human_train.head()
X_train=human_train.drop(['subject','Activity'],axis=1)
y_train=human_train.Activity
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

y_train=encoder.fit_transform(y_train)

y_test=encoder.fit_transform(y_test)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn import metrics
model=SVC(kernel='linear')

model.fit(X_train,y_train)
y_pred=model.predict(x_test)
print("accuracy: ", metrics.accuracy_score(y_true=y_test,y_pred=y_pred))
params={'kernel':['linear','rbf'],'C':[1,10,100],'gamma':[1e-2,1e-3,1e-4]}
classifier=GridSearchCV(estimator=SVC(),param_grid=params,scoring='accuracy',return_train_score=True)
classifier.fit(X_train,y_train)
classifier.best_params_
classifier.best_estimator_
svm_final=SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',

    max_iter=-1, probability=False, random_state=None, shrinking=True,

    tol=0.001, verbose=False)
svm_final.fit(X_train,y_train)
y_final_pred=svm_final.predict(x_test)
print("accuracy: ", metrics.accuracy_score(y_true=y_test,y_pred=y_final_pred))