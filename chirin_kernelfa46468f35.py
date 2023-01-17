# Import Data Processing Modules



import pandas as pd

import numpy as np

# Import Dataset and Show Head



df = pd.read_csv("../input/africa_recession.csv")

df.head()

# Define X and Y Variables



dependent_variable = 'growthbucket'



X = df[df.columns.difference([dependent_variable])]

Y = df[dependent_variable]

from sklearn.model_selection import train_test_split



train_features, val_features, train_targets, val_targets = train_test_split(X, Y, 

                                                                            test_size=0.30, 

                                                                            #stratify=Y,

                                                                            random_state=7)

mean = np.mean(train_features, axis=0)

train_features -= mean

val_features -= mean

std = np.std(train_features, axis=0)

train_features /= std

val_features /= std

# Load Basic ML Packages



from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

# Import the Random Forest Classifier from Sklearn



from sklearn.ensemble import RandomForestClassifier



# The class_weight Parameter is Set to "balanced" to Account for the Class Imbalance



ensemble = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',

                                  max_depth=1000, max_features='auto', max_leaf_nodes=10,

                                  min_impurity_decrease=0.0, min_impurity_split=None,

                                  min_samples_leaf=1, min_samples_split=2,

                                  min_weight_fraction_leaf=0.0, n_estimators=40000, n_jobs=None,

                                  oob_score=False, random_state=0, verbose=0, warm_start=False)

# Fit the Model to the Training Set and Test its Accuracy



ensemble.fit(train_features, train_targets)



predicted = ensemble.predict(val_features)



print('Accuracy achieved is: ' + str(np.mean(predicted == val_targets)))

print(metrics.classification_report(val_targets, predicted, target_names=("0", "1"))),

metrics.confusion_matrix(val_targets, predicted)
