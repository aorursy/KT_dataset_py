import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import svm

from matplotlib import style

style.use("ggplot")

%matplotlib inline
cells = pd.read_csv("../input/data.csv")
i = 0

for row in cells["diagnosis"]:

    if row == 'M':

        cells.set_value(i,'diagnosis', 1)

    else:

        cells.set_value(i, 'diagnosis', 0)

    i += 1



cells["diagnosis"] = pd.to_numeric(cells["diagnosis"], errors='coerce')

    
print(cells.head())
cells.diagnosis.value_counts().plot(kind='barh')
cells.describe()
cells.corr()
cells.corr()["diagnosis"].plot(kind="bar")

cells.corr()["diagnosis"]
from sklearn.model_selection import train_test_split
cells.columns
# This creates the classifier

clf = svm.SVC(kernel='linear', C = 1.0)
X = cells[['radius_mean', 'texture_mean', 'perimeter_mean',

          'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

          'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

          'radius_se', 'texture_se', 'perimeter_se', 'area_se','smoothness_se',

          'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

          'fractal_dimension_se','radius_worst','texture_worst',

          'perimeter_worst', 'area_worst', 'smoothness_worst',

          'compactness_worst', 'concavity_worst', 'concave points_worst',

          'symmetry_worst','fractal_dimension_worst']]

y = cells['diagnosis']



# Splits the data into traning and testing sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)



# This trains the classifier model

clf.fit(X_train,y_train)
# Import the scikit-learn function to compute eror

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score



# Generate our predictions for the test set.

predictions = clf.predict(X_test)



# Compute error between our test predictions and the actual value

mean_squared_error(predictions, y_test)

# Accuracy Score

print(accuracy_score(y_test, predictions))



print("{}%".format(round(accuracy_score(y_test, predictions)*100)))