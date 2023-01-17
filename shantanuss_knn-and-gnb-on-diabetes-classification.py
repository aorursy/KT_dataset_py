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
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score



from sklearn.model_selection import (KFold, StratifiedKFold,

                                     cross_val_predict, cross_val_score,

                                     train_test_split)



from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB



input_data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

input_data
input_data.isnull()
input_data.describe()
input_data.isin([0]).any()
from fancyimpute.knn import KNN





features_with_zero_values = ['BMI', 'BloodPressure', 'Glucose', 'Insulin', 'SkinThickness']

input_data[features_with_zero_values] = input_data[features_with_zero_values].replace(0, np.nan)

data = KNN(k=5).fit_transform(input_data.values)

inputdata = pd.DataFrame(data, columns=input_data.columns)
inputdata.isin([0]).any()  
inputdata.isna().any()  
inputdata.isnull().any()  
inputdata.hist(figsize=(12, 12))

plt.show()
X = inputdata.drop('Outcome', axis=1)   # input feature vector

y = inputdata['Outcome']                # labelled target vector



scaler = StandardScaler()                # scaling 

X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

X.head()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=10, stratify=y)

X_train.head()
y_train.head()
# cross validation kFold



kfold = StratifiedKFold(n_splits=10, random_state = 10)

# KNN model

clf = KNeighborsClassifier(n_neighbors=3)



# model fitting

clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=kfold)

scores.mean()
# prediction on validation data



y_pred = clf.predict(X_valid)

y_pred
# confusion matrix



cfm = confusion_matrix(y_pred, y_valid)

cfm

# accuracy score

print('accuracy of KNN: ',accuracy_score(y_pred,y_valid))
# model



clf_g = GaussianNB()

clf_g
# validation score



scores = cross_val_score(clf_g, X_train, y_train, cv=kfold)

scores.mean()
# model fitting

clf_g.fit(X_train, y_train)
# prediction on validation data



y_pred = clf_g.predict(X_valid)

y_pred
# confusion matrix



cfm = confusion_matrix(y_pred, y_valid)

cfm
# accuracy score

print('accuracy of GaussianNB: ',accuracy_score(y_pred,y_valid))