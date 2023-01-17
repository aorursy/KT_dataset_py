import numpy as np
import pandas as pd
dt = pd.read_csv('../input/winequality-red.csv')
dt.shape

dt.head()
dt.isna().sum()
dt.quality.min()
dt.quality.max()
for x in dt.columns:
    print(dt[[x,'quality']].corr())
    print()

import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(dt.quality,10,density=False, facecolor='blue')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
dt.columns
x = dt[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']].copy()
y= dt[['quality']].copy()
print(x.shape)
print(y.shape)
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.33, random_state=324)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

wine_quality_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=1)
wine_quality_classifier.fit(x_train,y_train)
predictions = wine_quality_classifier.predict(x_test)
predictions[:5]

accuracy_score(y_true=y_test, y_pred=predictions)