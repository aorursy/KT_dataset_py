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
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
# df.head() returns the first 5 rows of data. df.tail() will return the last 5 rows. 

print(df.head())
# Summarize the statistics of each column

df.describe()
# Plot a histogram of "age"

df["age"].hist()
# Drop the target, trestbps from features. We experimentally deterimine that trestbps is a weakly correlated feature to heart disease.

features = df.drop(["target","trestbps"],axis=1)

labels = df["target"]

print(labels.head())
#Normalize the data. Skip this on the first pass.

from sklearn.preprocessing import StandardScaler

std_scl = StandardScaler()

features = std_scl.fit_transform(features)
# Split up the data into training and testing sets. Model will be fit with X_train (features) and y_train (labels)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 4)
# Import two different classifiers to test. Knn and RF. 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)

# clf = KNeighborsClassifier(n_neighbors=3)



# Fit the model

clf.fit(X_train,y_train)
# Let's look at the feature importances, and which ones are most correlated with successful prediction: 

importances = []

names = list(df)

for i, imp in enumerate(clf.feature_importances_):

    importances.append( (imp, names[i]) )



importances = sorted(importances)

importances.reverse()

for v in importances:

    print("%s: %s" % (v[1],round(v[0],6)))
# Make predictions on the test feature set

preds = clf.predict(X_test)
# Assess the model in two different ways: confusion matrix and accuracy score.

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(preds,y_test)

acc = accuracy_score(preds,y_test)

print("Confusion matrix is \n %s" % cm)

print("Model accuracy is %s" % acc)