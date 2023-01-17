# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
index = 5

dataframe = pd.read_csv('../input/HR_comma_sep.csv')

dataframe = dataframe[['satisfaction_level','last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary', 'left']]

print(dataframe.head())

print(dataframe.shape)
X = dataframe.iloc[:,:-1].values

#X_orig = X.copy

X_orig = np.array(X)

print('X shape: ', X.shape)

y = dataframe.iloc[:,-1].values

print('y shape: ', y.shape)

print('Any missing values: ', dataframe.isnull().values.any())
# Categorical Features

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# Sales

label_encoding_X_sales = LabelEncoder()

X[:, 7] = label_encoding_X_sales.fit_transform(X[:, 7])



# Salary

label_encoding_X_salary = LabelEncoder()

X[:, 8] = label_encoding_X_salary.fit_transform(X[:, 8])



print('After Categorical\n', X[index])

print('After Categorical Shape: ', X.shape)
# One Hot Encoding

one_hot_encoder = OneHotEncoder(categorical_features = [7,8])

X = one_hot_encoder.fit_transform(X).toarray()

print('After OneHotEncoder\n', X[index])

print('After OneHotEncoder Shape: ', X.shape)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

print('After Scaling \n', X[index])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print

print('X_train: ',X_train.shape)

print('y_train: ',y_train.shape)

print('X_test: ',X_test.shape)

print('y_test: ',y_test.shape)
# Prediction

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)



y_train_pred = classifier.predict(X_train)

print('Train Accuracy Score: ', accuracy_score(y_train_pred, y_train))



y_pred = classifier.predict(X_test)

print('Test Accuracy Score: ', accuracy_score(y_pred, y_test))
'''

# Single prediction

p_index = 10

print(X_orig[p_index])

print(y_train[p_index])



print('\nPrediction\n')

#X_test_prediction = np.array([[0.45, 0.54, 2, 135, 3, 0, 0, 'sales', 'low']])

X_test_prediction = np.array(X_orig[p_index])

print(X_test_prediction)



X_test_prediction[:, 7] = label_encoding_X_sales.transform(X_test_prediction[:, 7])

X_test_prediction[:, 8] = label_encoding_X_salary.transform(X_test_prediction[:, 8])

X_test_prediction = one_hot_encoder.transform(X_test_prediction).toarray()

X_test_prediction = sc.fit_transform(X_test_prediction)



y_test_prediction = classifier.predict(X_test_prediction)

print(y_test_prediction)

'''