!pip install treeplot



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import treeplot package:

import treeplot

# and the random forest classifier

from sklearn.ensemble import RandomForestClassifier



# read in the data

train_data = pd.read_csv('../input/titanic/train.csv')



# select some features

features = ["Pclass", "Sex", "SibSp", "Parch"]



X_train       = pd.get_dummies(train_data[features])

y_train       = train_data["Survived"]



# perform the classification and the fit

classifier = RandomForestClassifier(criterion='gini', n_estimators=100, 

        min_samples_split=2, min_samples_leaf=10, max_features='auto')

classifier.fit(X_train, y_train)
# now make the plot

ax = treeplot.plot(classifier)