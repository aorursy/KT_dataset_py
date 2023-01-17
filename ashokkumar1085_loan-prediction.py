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
# Importing the necessary Libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



sb.set_style("whitegrid")
# Importing the Dataset

loan_train = pd.read_csv("/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")

loan_test = pd.read_csv("/kaggle/input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv")
loan_train.columns
loan_test.columns
loan_train.head()
# Checking for duplicate records

loan_train.duplicated().sum()
# Dropping Unwanted columns

loan_train.drop(axis = 1, columns = "Loan_ID", inplace = True)
# Splitting the train dataset into dependent and Independent Variable



X = loan_train.iloc[:, :-1].values

y = loan_train.iloc[:, -1].values
X[:5, :]
# Checking for Null values

pd.isnull(X[:, [0]]).sum()
cols = loan_train.columns
for i in cols:

    print(i)

    print("Total Null Values\t:\t", loan_train[i].isnull().sum())

    print(loan_train[i].value_counts())

    print()
# Filling the Null values



from sklearn.impute import SimpleImputer



gender_imputer = SimpleImputer(strategy = "most_frequent")

X[:, [0]] = gender_imputer.fit_transform(X[:, [0]])



marry_imputer = SimpleImputer(strategy = "most_frequent")

X[:, [1]] = marry_imputer.fit_transform(X[:, [1]])



dependents_imputer = SimpleImputer(strategy = "most_frequent")

X[:, [2]] = dependents_imputer.fit_transform(X[:, [2]])



employ_imputer = SimpleImputer(strategy = "most_frequent")

X[:, [4]] = employ_imputer.fit_transform(X[:, [4]])



LA_imputer = SimpleImputer(strategy = "mean")

X[:, [7]] = LA_imputer.fit_transform(X[:, [7]])



LAT_imputer = SimpleImputer(strategy = "most_frequent")

X[:, [8]] = LAT_imputer.fit_transform(X[:, [8]])



cred_imputer = SimpleImputer(strategy = "most_frequent")

X[:, [9]] = cred_imputer.fit_transform(X[:, [9]])
pd.Series(X[:, 10]).value_counts()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer



gender_encoder = LabelEncoder()

X[:, 0] = gender_encoder.fit_transform(X[:, 0])



marry_encoder = LabelEncoder()

X[:, 1] = marry_encoder.fit_transform(X[:, 1])







graduate_encoder = LabelEncoder()

X[:, 3] = graduate_encoder.fit_transform(X[:, 3])



self_emp_encoder = LabelEncoder()

X[:, 4] = self_emp_encoder.fit_transform(X[:, 4])







column_transformer = ColumnTransformer(transformers = [('PA_encoder', OneHotEncoder(), [10])], remainder="passthrough")

X = column_transformer.fit_transform(X)
X[:5, :]
print(X[:, 5])
def dependents_column_changer(a):

    if a == "0":

        return 0

    elif a == "1":

        return 1

    elif a == "2":

        return 2

    else:

        return 3
X[:, 5] = [dependents_column_changer(a) for a in X[:, 5]]
X[:10, :]
y[:10]
sb.relplot(x = "LoanAmount", y = "ApplicantIncome", data = loan_train, hue = "Loan_Status", col = "Gender")
X[:5, :]
corr = loan_train.corr()

corr
sb.heatmap(data = corr)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = "liblinear")
classifier.fit(X, y)
classifier.predict([[1.0, 0.0, 0.0, 1, 0, 0, 0, 0, 5849, 0.0, 146.41216216216216, 360.0, 1.0]])
loan_test.head()
loan_test.drop(axis = 1, columns = "Loan_ID", inplace = True)
X_test = loan_test.iloc[:, :].values


X_test[:, [0]] = gender_imputer.transform(X_test[:, [0]])



X_test[:, [1]] = marry_imputer.transform(X_test[:, [1]])



X_test[:, [2]] = dependents_imputer.transform(X_test[:, [2]])



X_test[:, [4]] = employ_imputer.transform(X_test[:, [4]])



X_test[:, [7]] = LA_imputer.transform(X_test[:, [7]])



X_test[:, [8]] = LAT_imputer.transform(X_test[:, [8]])



X_test[:, [9]] = cred_imputer.transform(X_test[:, [9]])
pd.isnull(X_test).sum()
X_test[:, 0] = gender_encoder.transform(X_test[:, 0])



X_test[:, 1] = marry_encoder.transform(X_test[:, 1])



X_test[:, 3] = graduate_encoder.transform(X_test[:, 3])



X_test[:, 4] = self_emp_encoder.transform(X_test[:, 4])



X_test = column_transformer.transform(X_test)
X_test[:, 5] = [dependents_column_changer(a) for a in X_test[:, 5]]
X_test[:1, :]
# Predicting the test Data



y_pred = classifier.predict(X_test)
y_pred