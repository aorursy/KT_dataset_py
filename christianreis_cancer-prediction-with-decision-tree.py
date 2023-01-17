import numpy as np, matplotlib.pyplot as plt, pandas as pd

from collections import Counter
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
dataset = pd.read_csv('../input/kag_risk_factors_cervical_cancer.csv', header=0, na_values='?')
dataset.head(10)
dataset.isnull().sum()
dataset['Biopsy'].value_counts()
dataset.drop(dataset.columns[[26, 27]], axis=1, inplace=True)
values = dataset.values
X = values[:, 0:33]
y = values[:, 33]
imputer = Imputer(strategy='median')
X = imputer.fit_transform(X)
iht = InstanceHardnessThreshold(random_state=12)
X, y = iht.fit_sample(X, y)
print('Amount of each class after under-sampling: {0}'.format(Counter(y)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12)
classifier = DecisionTreeClassifier(criterion='gini', max_leaf_nodes= None,
                                    min_samples_leaf=14, min_samples_split=2 ,
                                    random_state = 12)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted No', 'Predicted Yes'],
    index=['Actual No', 'Actual Yes']
)
scores = cross_val_score(classifier, X, y, scoring='f1_macro', cv=10)
print('Macro-F1 average: {0}'.format(scores.mean()))