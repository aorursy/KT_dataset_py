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
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
#splitting the training set into training and validation subsets



from sklearn.model_selection import train_test_split



# Split into validation and training data, set to random_state 1

train_set, validation_set = train_test_split(train, test_size = 0.20, random_state = 1)



y_train = train_set.Cover_Type

y_validation = validation_set.Cover_Type



train_set.drop(['Cover_Type'], axis = 1, inplace = True)

validation_set.drop(['Cover_Type'], axis = 1, inplace = True)
#. choosing classifier

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0)

classifier.fit(train_set, y_train)

y_train_predicted_basic = classifier.predict(train_set)

y_validation_predicted_basic = classifier.predict(validation_set)
#checking accuracy



from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score





scores_rf_training_basic = cross_val_score(classifier, train_set, y_train, cv=10, scoring='accuracy')

accuracy_training_basic = accuracy_score(y_train_predicted_basic,y_train)



scores_rf_validation_basic = cross_val_score(classifier, validation_set, y_validation, cv=10, scoring='accuracy')

accuracy_validation_basic = accuracy_score(y_validation_predicted_basic,y_validation)

# Get the mean accuracy score



print("cross_val_score on basic RF (train): ", scores_rf_training_basic.mean())

print("cross_val_score on basic RF (validation): ", scores_rf_validation_basic.mean())

print("accuracy score train: ", accuracy_training_basic)

print("accuracy score validation: ", accuracy_validation_basic)



classifier.get_params().keys()
# optimizing the parameters - Grid Search 

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

"""

from sklearn.model_selection import GridSearchCV

parameters = [{'n_estimators': [1,10,100,200,300,400]},

             {'n_estimators': [1,10,100,200,300,400],'max_features': ['sqrt']},

             {'n_estimators': [1,10,100,200,300,400], 'bootstrap': [False],'max_features': ['sqrt']},

             {'n_estimators': [1,10,100,200,300,400],'max_features': ['log2']},

             {'n_estimators': [1,10,100,200,300,400], 'bootstrap': [False],'max_features': ['log2']}

             ]

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, cv = 10, n_jobs = -1)

grid_search = grid_search.fit(train_set, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("best_accuracy: ", best_accuracy, "\nbest_parameters: ", best_parameters)"""

# make predictions using the model



final_classifier = RandomForestClassifier(bootstrap = False, max_features = 'sqrt', n_estimators = 300, random_state = 0)



final_classifier = final_classifier.fit(train_set, y_train)

predictions_test = final_classifier.predict(test)



#Gradient Boosting - XGBoost classifier

"""X_train = train_set

y_train = y_train

X_test = validation_set

y_test = y_validation



from xgboost import XGBClassifier

import time



xgb = XGBClassifier(n_estimators=100)

training_start = time.perf_counter()

xgb.fit(X_train, y_train)

training_end = time.perf_counter()

prediction_start = time.perf_counter()

preds = xgb.predict(X_test)

prediction_end = time.perf_counter()

acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100

xgb_train_time = training_end-training_start

xgb_prediction_time = prediction_end-prediction_start

print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb)) #76.49

print("Time consumed for training: %4.3f" % (xgb_train_time)) #14.595

print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time)) #0.07857 seconds"""



#Permutation importance

# https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html



import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(final_classifier, random_state = 1).fit(train_set,y_train)

eli5.show_weights(perm, feature_names = train_set.columns.tolist())



"""

Weight 	Feature

0.2315 ± 0.0043 	Elevation

0.0435 ± 0.0025 	Horizontal_Distance_To_Roadways

0.0229 ± 0.0010 	Id

0.0162 ± 0.0011 	Horizontal_Distance_To_Hydrology

0.0138 ± 0.0019 	Horizontal_Distance_To_Fire_Points

0.0108 ± 0.0002 	Soil_Type3

0.0101 ± 0.0008 	Soil_Type10

0.0065 ± 0.0006 	Soil_Type39

0.0051 ± 0.0009 	Soil_Type38

0.0030 ± 0.0002 	Soil_Type4

0.0018 ± 0.0004 	Wilderness_Area4

0.0014 ± 0.0004 	Vertical_Distance_To_Hydrology

0.0006 ± 0.0003 	Hillshade_9am

0.0005 ± 0.0004 	Aspect

0.0004 ± 0.0002 	Hillshade_Noon

0.0003 ± 0.0001 	Soil_Type35

0.0003 ± 0.0001 	Soil_Type2

0.0002 ± 0.0000 	Soil_Type12

0.0002 ± 0.0002 	Soil_Type40

0.0002 ± 0.0001 	Soil_Type30

… 35 more …

"""
from sklearn.feature_selection import SelectFromModel



# ... load data





# perm.feature_importances_ attribute is now available, it can be used

# for feature selection - let's e.g. select features which increase

# accuracy by at least 0.01:

sel = SelectFromModel(perm, threshold=0.01, prefit=True)

X_trans = sel.transform(train_set)

#creating output file

output = pd.DataFrame({'Id': test.Id,

                       'Cover_Type': predictions_test})

output.to_csv('rf_basic.csv', index=False)
