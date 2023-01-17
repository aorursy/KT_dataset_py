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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score

from mlxtend.classifier import StackingCVClassifier

from sklearn.model_selection import cross_validate
import pandas as pd

train = pd.read_csv('../input/learn-together/train.csv', index_col = 'Id')

test = pd.read_csv('../input/learn-together/test.csv', index_col = 'Id')

train.columns

train.info()

train.head()
# Make a copy of train df for ML experiments

train_2 = train.copy()

train_2.columns
# Separate feature and target arrays as X and y

X = train_2.drop('Cover_Type', axis = 1)

y=train_2.Cover_Type

print(X.columns)

y[:5]
# Split X and y into Train and Validation sets

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2,random_state = 99)
# Plug in Random Forest Best Estimator Parameters

rf = RandomForestClassifier(n_estimators = 1930, 

                            min_samples_split = 5, 

                            min_samples_leaf = 1, 

                            max_features = 0.3, 

                            max_depth = 46, 

                            bootstrap = False,

                            random_state=42)

rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_val)

print('Random Forest Best Estimator validation set accuracy is: ', accuracy_score(y_val,y_pred_rf))
# Plug in Extra Trees Best Estimator Parameters

extra_trees = ExtraTreesClassifier(n_estimators = 3162,

                                   min_samples_split = 5, 

                                   min_samples_leaf = 1, 

                                   max_features = 0.5,

                                   max_depth = 464, 

                                   bootstrap = False,

                                   random_state=42)

extra_trees.fit(X_train,y_train)

y_pred_extra = extra_trees.predict(X_val)

print('Extra Trees Best Estimator validation set accuracy is: ', accuracy_score(y_val,y_pred_extra))
# Plug in LightGBM Best Estimator Parameters

lgbm = LGBMClassifier(random_state = 42,

                      n_estimators = 268, 

                      min_samples_split = 5, 

                      min_data_in_leaf = 1, 

                      max_depth = 21, 

                      learning_rate = 0.05, 

                      feature_fraction = 0.5, 

                      bagging_fraction = 0.5 , 

                      is_training_metric = True)

lgbm.fit(X_train,y_train)

y_pred_lgbm = lgbm.predict(X_val)

print('LightGBM Best Estimator validation set accuracy is: ', accuracy_score(y_val,y_pred_lgbm))
# Plug in LightGBM 'Manual' Potential Best Estimator parameters

lgbm = LGBMClassifier(random_state=42,

                      n_estimators=3162, 

                      min_samples_split=5, 

                      min_data_in_leaf=1, 

                      max_depth=21, 

                      learning_rate=0.05,

                      feature_fraction=0.9, 

                      bagging_fraction=0.6 , 

                      is_training_metric = True)

lgbm.fit(X_train,y_train)

y_pred_lgbm = lgbm.predict(X_val)

print('LightGBM Best Estimator validation set accuracy is: ', accuracy_score(y_val,y_pred_lgbm))
stack = StackingCVClassifier(classifiers=[rf,

                                         extra_trees,

                                         lgbm],

                            use_probas=True,

                            meta_classifier=extra_trees, random_state = 42)



stack.fit(X_train,y_train)

print('Stacking Classifier Cross-Validation accuracy scores are: ',cross_val_score(stack,X_train,y_train, cv = 5))

y_pred_stack = stack.predict(X_val)

print('Stacking Classifier validation set accuracy is: ', accuracy_score(y_val,y_pred_stack))

# Feature Importances from Random Forest Best Estimator

rf_importance = pd.DataFrame(list(zip(X_train.columns, list(rf.feature_importances_))), columns = ['Feature', 'Importance'])

rf_importance.sort_values('Importance', ascending = False)
# Feature Importances from Extra Trees Best Estimator

extra_importance = pd.DataFrame(list(zip(X_train.columns, list(extra_trees.feature_importances_))), columns = ['Feature', 'Importance'])

extra_importance.sort_values('Importance', ascending = False)
# Feature Importances from LGBM Best Estimator

lgbm_importance = pd.DataFrame(list(zip(X_train.columns, list(lgbm.feature_importances_))), columns = ['Feature', 'Importance'])

lgbm_importance.sort_values('Importance', ascending = False)