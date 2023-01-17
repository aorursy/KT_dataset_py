# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

df.head()
# Diagnosis (M = malignant, B = benign) , qualitative dependent variable. can use logistic regression
# no missing values
df.info()
# ID number : 1st column can be done away with in analysis
# last column as well as only has null value

X = df.iloc[: , 2:31].values
y = df.iloc[: , 1].values
#splitting the dataset into train and test data

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , train_size = 0.8)
#standardising the independent variables

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#implementing the logistic regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train , y_train)
#predicting the dependent variable

y_pred = classifier.predict(X_test)
#evaluating model's performance

from sklearn.metrics import  confusion_matrix , accuracy_score
confusion_matrix(y_test , y_pred)

accuracy_score(y_test , y_pred)
#cross validating results
#partitioning the data into k = 5 (folds). Then , randomly  training the algorithm on 4(k-1) 
# folds while using the remaining fold as the test set .

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier ,X = X_train , y=y_train , cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))





