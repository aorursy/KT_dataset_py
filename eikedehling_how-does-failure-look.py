import numpy as np

import pandas as pd



hdd = pd.read_csv('../input/harddrive.csv')



hdd.shape
hdd.head()
import seaborn as sns



print(hdd.groupby('failure').size())



sns.countplot(x="failure", data=hdd)
## Drop any constant-value columns

## Takes too long :-(

#for i in hdd.columns:

#    if len(hdd.loc[:,i].unique()) == 1:

#        hdd.drop(i, axis=1, inplace=True)



# Drop the normalized columns..

hdd = hdd.select(lambda x: x[-10:] != 'normalized', axis=1)



hdd.shape
X = hdd.drop(['date', 'serial_number', 'model', 'capacity_bytes', 'failure'], axis=1)[:100000]

y = hdd['failure'][:100000]



X.fillna(value=0, inplace=True)
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier



##

## Commented because it runs long.. this finds good class-weights to deal with the imbalanced

## class distribution..

##

# gsc = GridSearchCV(

#      estimator=DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=20),

#      param_grid={

#          'class_weight': [{0: 1, 1: x} for x in range(150, 251, 25)]

#      },

#      scoring='f1',

#      cv=5

# )

#

# grid_result = gsc.fit(X, y)

#

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)



tree = DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=20, class_weight={0: 1, 1: 175})

tree.fit(X, y)

print (classification_report(y, tree.predict(X)))
for feat, imp in zip(X.columns, tree.feature_importances_):

    if imp > 0.0001:

        print("- %s  => %.3f" % (feat, imp))
one_drive = hdd[hdd['serial_number'] == 'S30114J3']



one_drive['smart_197_raw'].plot()

one_drive['smart_198_raw'].plot()

one_drive['failure'].plot()