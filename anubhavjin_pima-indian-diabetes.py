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
data=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

data.shape
data.info
data.Outcome.value_counts()
data.head()
data.isnull().values.any()
corrmat=data.corr()

top_corr_features=corrmat.index

g=sns.heatmap(data[top_corr_features].corr(),annot=True)
data.corr()
from sklearn.model_selection import train_test_split

feature_columns=['Pregnancies',	'Glucose'	,'BloodPressure',	'SkinThickness'	,'Insulin'	,'BMI'	,'DiabetesPedigreeFunction'	,'Age']

predicted_features=['Outcome']
X=data[feature_columns].values

y=data[predicted_features].values



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)
from sklearn.impute import SimpleImputer

fill_values=SimpleImputer(missing_values=0,strategy='mean')

X_train=fill_values.fit_transform(X_train)

X_test=fill_values.fit_transform(X_test)
X_train.head()
from sklearn.ensemble import RandomForestClassifier

random_forest_model=RandomForestClassifier(random_state=1)

random_forest_model.fit(X_train,y_train.ravel())
output=random_forest_model.predict(X_test)

from sklearn import metrics



print('accuracy= {0:.3f}'.format(metrics.accuracy_score(y_test,output)))
params={

    'learning_rate' :[0.05,0.1,0.15,0.2,0.25,0.3],

    'max_depth':[3,5,7,9,10,12,15],

    'min_child_weight':[1,3,5,7],

    'gamma':[0.0,0.1,0.2,0.3,0.4],

    'colsample_bytree':[0.3,0.4,0.5,0.7]

}
from sklearn.model_selection import RandomizedSearchCV

import xgboost
classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_train,y_train.ravel())
random_search.best_estimator_
classifier=xgboost.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.5, gamma=0.0, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.1, max_delta_step=0, max_depth=10,

              min_child_weight=7, missing=None, monotone_constraints=None,

              n_estimators=100, n_jobs=0, num_parallel_tree=1,

              objective='binary:logistic', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

              validate_parameters=False, verbosity=None)
classifier.fit(X_train,y_train.ravel())
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score



cm=confusion_matrix(y_test,y_pred)

score=accuracy_score(y_test,y_pred)

print(cm,end='\n')

print(score)
from sklearn.model_selection import cross_val_score

score=cross_val_score(classifier,X_train,y_train.ravel(),cv=10)
score

score.mean()