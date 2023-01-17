# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import sys
import pandas as pd
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
# Start with a simple model first and then refine
# Step 1: EDA
# Import training data with pandas in a dataframe
path_training = r"../input/train.csv"
path_testing = r"../input/test.csv"
path_answers = r"submission.csv"

# Print an example of the training data (head or tail)
df_training = pd.read_csv(path_training)
df_testing = pd.read_csv(path_testing)
print(df_training.head())
print(df_testing.head())
# Remove features that does not impact the target values
# Remove name and ticket. Cabin is dropped but could be relevant since it could give some information on the position of the cabin on the ship and the position of the cabin can affect 
# where the passengers were at the time of the accident
df_train = df_training.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test = df_testing.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_train.head(20)
df_test.head(20)
X_train_temp = df_train.drop(['Survived'], axis=1)
X_test_temp = df_test # No need to drop the survived column since the feature is not in the .csv file
X_train_temp.head(20)
X_test_temp.head(20)
# Choose a model with sklearn depending on the quantity of data
# We are doing classification and we have a less than 1k of records so the SVC model from sklearn will be where we will start
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(max_depth=2, random_state=0)
# clf = LinearSVC(C=0.07, random_state=0)
# clf = KNeighborsClassifier(n_neighbors=3)
X_train_temp.isnull().any()
X_test_temp.isnull().any()
X_train_temp = X_train_temp.fillna(X_train_temp.mean())
X_test_temp = X_test_temp.fillna(X_test_temp.mean())
X_train_temp.isnull().any() # Bug to fix: Embarked has NA value but cannot see it in csv file
X_test_temp.isnull().any()
# Selecting the target vector and transforming it as a numpy array.
y = df_train['Survived'].as_matrix()
y.size
print(type(y))
print(type(df_train['Survived']))
X_train_one_hot_encoded = pd.get_dummies(X_train_temp)
X_test_one_hot_encoded = pd.get_dummies(X_test_temp)
plt.matshow(X_train_one_hot_encoded.corr())
X = X_train_one_hot_encoded.as_matrix()
X_test = X_test_one_hot_encoded.as_matrix()
X.shape, y.shape
# Split dataset for cross validation

from sklearn.cross_validation import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.7)
print(X_train.shape, y_train.shape)
clf.fit(X_train, y_train)
predictions = clf.predict(X_validation)
from sklearn.metrics import accuracy_score
accuracy_score(y_validation, predictions)
survived = list(clf.predict(X_test))
passenger_id = list(df_test['PassengerId'])
# Create dataframe with passengerId and predictions and export it in csv
answers = {'PassengerId': passenger_id,
           'Survived': survived}
answers_to_submit = pd.DataFrame(data=answers)
answers_to_submit.head()
answers_to_submit.to_csv(path_answers)
