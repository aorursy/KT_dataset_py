# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/diabetes.csv")

data.head()
#Split the data into variables and outcome

X = data[data.columns[1:8]] #The variables

X = X.drop('DiabetesPedigreeFunction', 1)

Y = data[data.columns[8]] #The outcome

X.head()
#Split the data randomly into two. One for training the model, one for testing

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size= .5)
#Make and fit the random forest to the training data

randForest = RandomForestClassifier(n_estimators=10, max_features = 2.2, min_samples_leaf = 3)

randForest = randForC.fit(X_train, Y_train) 
params = randForest.predict(X_test)

#Accuracy of the predictions

randForest.score(X_test,Y_test)
#Find the importance of each feature

feats = randForest.feature_importances_

vars = ['Glocose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

print(vars)

print(feats)
#Plot the importance of each feature

fig = plt.figure()

ax = fig.add_subplot(111)

ax.barh(range(0,6), feats, align = 'center')

ax.set_yticks(range(0,6))

ax.set_yticklabels(vars)

ax.set_title("Feature Importance in Determining Diabetes")

fig.tight_layout()