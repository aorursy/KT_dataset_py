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
# Importing the datasets
training_set = pd.read_csv("../input/train.csv")
test_set = pd.read_csv("../input/test.csv")
training_set.head(20)
test_set.head(20)
# Display the number of missing values for each feature(column)
training_set.isnull().sum()
test_set.isnull().sum()
# Create Matrix of Features(Training set)
X_train = training_set.iloc[:, [2,4,5,6,7]].values # Pclass, Sex, Age, SibSp, Parch

# Create Dependent Variable Vector(Training set)
y_train = training_set.iloc[:, 1].values # Survived

# Create Matrix of Features(Test set)
X_test = test_set.iloc[:, [1,3,4,5,6]].values # Pclass, Sex, Age, SibSp, Parch

# No y_test as I'm not given the "survived" feature in the test set as this is a competition

# Take care of missing values in Age for both Training Set and Test Set
from sklearn.impute import SimpleImputer
imputer_train = SimpleImputer(strategy = 'mean')
imputer_test = SimpleImputer(strategy = 'mean')

# Training Set
imputer_train = imputer_train.fit(X_train[:, 2:3]) # Fit Age column to imputer
X_train[:, 2:3] = imputer_train.transform(X_train[:, 2:3]) # Convert NaN's to mean of whole column

# Test Set
imputer_test = imputer_test.fit(X_test[:, 2:3]) # Fit Age column to imputer
X_test[:, 2:3] = imputer_test.transform(X_test[:, 2:3]) # Convert NaN's to mean of whole column

# Couldn't easily find a built in numpy way to compute this so I just made my own function to count how many NaN's are in the imputed X_train ndarray
def numMissing(X_train):
    num_nans = 0
    for y in X_train:
        if y[2] == np.nan:
            count = count + 1
    return num_nans

print("Training Set: Number of missing values in age: {}".format(numMissing(X_train)))
print("Test Set: Number of missing values in age: {}".format(numMissing(X_test)))

X_train
X_test
# Encoding categorical data for X_train
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encode Sex feature in X_train and X_test
labelencoder_train_sex = LabelEncoder()
X_train[:, 1] = labelencoder_train_sex.fit_transform(X_train[:, 1])
labelencoder_test_sex = LabelEncoder()
X_test[:, 1] = labelencoder_test_sex.fit_transform(X_test[:, 1])

# OneHotEncode X_train
onehotencoder_train = OneHotEncoder(n_values = 'auto', categories = 'auto')
X_train = onehotencoder_train.fit_transform(X_train[:, [0,1,3,4]]).toarray()

# OneHotEncode X_test
onehotencoder_test = OneHotEncoder(n_values = 'auto', categories = 'auto')
X_test = onehotencoder_test.fit_transform(X_test[:, [0,1,3,4]]).toarray()

# Print shape
print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_train
X_test
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_set["PassengerId"],
    "Survived": y_pred
})
submission.shape
submission
submission.to_csv('survive_or_not.csv')
print(os.listdir("../working"))
